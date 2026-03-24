import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
from fixermodule import fix_torch_import_issue, fix_inject_top_k_p_filtering
import torch.nn.functional as F 
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts
import torch.optim as optim 
import config 
from config import LRMODE
from CVUSA_Manager import CVUSADataset
from taming_interface import download_taming_vqgan, save_checkpoint,manual_forward_pass, getDevice, build_model, get_optimizer
import wandb

def train_one_epoch(model, train_dataloader, optimizer, scaler, device, tmask_pkeep=1.0):
    """Train for one epoch and return average loss"""
    model.train()           #Set Training mode
    epoch_loss = 0          #Starting Loss
    num_batches = 0         #Starting Batch?
    
    for batch_idx, batch in enumerate(train_dataloader):
        # Move to device, a batch,
        ground_imgs = batch['ground'].to(device)
        satellite_imgs = batch['satellite'].to(device)
        
        # Forward pass with mixed precision
        with autocast():
            logits, target = manual_forward_pass(model, satellite_imgs, ground_imgs, tmasking_pkeep=tmask_pkeep)    #ESEGUO FORWARD PASSS
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1),label_smoothing=0.1)     #CALCOLO LA LOSS
        
        # Backward pass with gradient scaling
        optimizer.zero_grad()
        scaler.scale(loss).backward()                                       #BACKWARD PASS
        
        scaler.unscale_(optimizer)                                          #REAL VALUE OBTAINING, I NEED THAT BECAUSE I AM USING GRADIENT CLIPPING
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)    #GRADIENT CLIPPING
        
        scaler.step(optimizer)                                              #CALL OPTIMIZER STEP
        scaler.update()                                                     
        
        # Track loss
        epoch_loss += loss.item()
        num_batches += 1
        
        # Print progress every 50 batches
        if batch_idx % 1500 == 0:
            print(f"  Batch {batch_idx}, kLoss: {loss.item():.4f}")
    
    return epoch_loss / num_batches

def evaluate_model(model, test_dataloader, device):
    """
    Evaluate model on test set and return metrics
    """
    model.eval()                    #SET MODEL TO EVALUATION
    #ACCUMULATION
    total_loss = 0
    total_tokens = 0
    correct_top1 = 0
    correct_top10 = 0
    num_batches = 0
    
    with torch.no_grad():   
        for batch in test_dataloader:   
            # Move to device
            ground_imgs = batch['ground'].to(device)
            satellite_imgs = batch['satellite'].to(device)
            
            # Forward pass
            logits, target = manual_forward_pass(model, satellite_imgs, ground_imgs)
            
            # Calculate loss
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), target.reshape(-1))
            
            # Calculate token accuracy
            predictions = logits.argmax(dim=-1)               #ARGMAX FOR EVERY TOKEN
            correct_top1 += (predictions == target).sum().item()

            top10_indices = torch.topk(logits, k=10, dim=-1).indices  # (B, T, 10)
            correct_top10 += (top10_indices == target.unsqueeze(-1)).any(dim=-1).sum().item()
            
            # Accumulate metrics
            total_loss += loss.item()                          
            total_tokens += target.numel()
            num_batches += 1

            if num_batches % 1500 == 0 or num_batches == total:
                print(f"  Eval {num_batches}/{total}")
    
    # Calculate averages metrics and statistics
    avg_loss = total_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss)).item()

    model.train()   # Switch back to training mode
    
    return {
        'loss': avg_loss,
        'perplexity': perplexity,
        'top1_accuracy': correct_top1 / total_tokens,
        'top10_accuracy': correct_top10 / total_tokens,
        'total_tokens': total_tokens,
        'total_batches': num_batches
    }

def train_model_with_evaluation(model, train_dataloader, test_dataloader, num_epochs=50, lr=5e-4, optimizer=None):
    """
    Modified training with train/test split and overfitting detection
    """
    # Setup training parameters with weight decay
    device = getDevice()
    model = model.to(device)
    #model = torch.compile(model)  # Can give 10-20% speedup
    
    # Use given optimizer or create a new one
    if optimizer is None:
        optimizer = get_optimizer(model, lr, weight_decay=0.1)


    
    scaler = GradScaler()                                                                               #create the scaler
    match config.LEARNING_RATE_MODE:
        case LRMODE.COSINEANNEALING:
            scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)                            #Cosine Annealing for LR Scheduling
        case LRMODE.COSINEANNEALING_WR:
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0 = 10, T_mult=1, eta_min=1e-6)                      #Cosine Annealing for LR Scheduling
        case LRMODE.FIXED:
            pass
    
    if config.USE_WANDB:
        if config.KAGGLE_FLAG:
            import kaggle_secrets
            os.environ["WANDB_API_KEY"] = kaggle_secrets.UserSecretsClient().get_secret("WANDB_API_KEY")
        else:
            from dotenv import load_dotenv
            load_dotenv()
        wandb.login()
        wandb.init(
            project="Ground2Satellite",
            config={
                "Total Epochs": num_epochs,
                "lr": lr,
                "Learning Rate Scheduling": config.LEARNING_RATE_MODE.name,
                "batch_size": config.BATCH_SIZE,
                "layers": config.LAYERS,
                "heads": config.HEADS,
                "dropout": config.DROPOUT,
                "Token Masking initial %": config.TOKEN_MASKING_SCHEDULING_START,
                "Token Masking final %": config.TOKEN_MASKING_SCHEDULING_END,
                "Training Set Batches num": len(train_dataloader),
                "Test Set Batches num": len(test_dataloader)
            }
        )
    
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Training set: {len(train_dataloader)} batches")
    print(f"Test set: {len(test_dataloader)} batches")
    print(f"Learning rate: {lr}, Weight decay: 0.01")
    print(f"Token masking: {config.TOKEN_MASKING_SCHEDULING_START}, to {config.TOKEN_MASKING_SCHEDULING_END}, in {num_epochs}")
    print(f"Training on device: {device}")
    
    # Tracking variables
    best_test_loss = float('inf')
    best_top10_retrieval= float('-inf')
    previous_gap = 0
    best_model_path = None  #Track the last saved best model to delete it
    
    for epoch in range(num_epochs):
        saved=False
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        current_token_masking = config.TOKEN_MASKING_SCHEDULING_START + (epoch / num_epochs) * (config.TOKEN_MASKING_SCHEDULING_END - config.TOKEN_MASKING_SCHEDULING_START)
        
        # TRAINING PHASE
        print("Training phase...")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, scaler, device, tmask_pkeep=current_token_masking) #LAUNCH TRAINING FOR THIS EPOCH
        
        # EVALUATION PHASE  
        print("Evaluation phase...")
        test_metrics = evaluate_model(model, test_dataloader, device)                               #EVALUATE FOR THIS EPOCH
        test_loss = test_metrics['loss']
        top1acc = test_metrics['top1_accuracy']
        top10acc = test_metrics['top10_accuracy']
        perp=test_metrics['perplexity']

        
        # UPDATE LEARNING RATE
        if config.LEARNING_RATE_MODE != LRMODE.FIXED:
            scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # OVERFITTING DETECTION (using absolute value)
        current_gap = abs(test_loss - train_loss)  
        gap_status = ""
        if epoch > 0:  # Skip first epoch comparison
            if current_gap > previous_gap:
                gap_status = "⚠️  WARNING: Gap widening (potential overfitting!)"
            else:
                gap_status = "✅ Gap stable/improving"
        
        previous_gap = current_gap
        
        #PRINT EPOCH SUMMARY
        print(f"\nEPOCH {epoch + 1} SUMMARY:")
        print(f"    Train Loss:     {train_loss:.4f}")
        print(f"    Test Loss:      {test_loss:.4f}")
        print(f"    Top1 Accuracy:  {top1acc:.3f} ({top1acc*100:.1f}%)")
        print(f"    Top10 Accuracy:  {top10acc:.3f} ({top10acc*100:.1f}%)")
        print(f"    Perplexity:  {perp:.2f}")
        print(f"    Learning Rate:  {current_lr:.2e}")
        print(f"    Loss Gap (abs): {current_gap:.4f}")

        if config.USE_WANDB:
            wandb.log({
                "train_loss": train_loss,
                "test_loss": test_loss,
                "top1_accuracy": top1acc,
                "top10_accuracy": top10acc,
                "perplexity": perp,
                "learning_rate": current_lr,
                "loss_gap": current_gap,
                "epoch": epoch + 1,
                "token_masking": current_token_masking,
            })

        if gap_status:
            print(f"   {gap_status}")
        
        # BEST MODEL SAVING (when test loss improves)
        if (test_loss < best_test_loss) and saved==False:
            # Calculate improvement
            improvement_amount = best_test_loss - test_loss
            improvement = f"improved LOSS by {improvement_amount:.4f}" if epoch > 0 else "first save"
            
            # Delete previous best model if it exists
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"-- Deleted previous best model: {os.path.basename(best_model_path)}")
            
            # Update best loss and save new best model
            best_test_loss = test_loss
            print(f"-- New best test loss! Saving 'improve' model... ({improvement})")
            best_model_path = save_checkpoint(model, optimizer, epoch+1, test_loss, 
                                    base_name="CVUSAGround2Satellite_improved")
            saved=True
        
        if (top10acc > best_top10_retrieval) and saved==False:
            # Calculate improvement
            improvementt10_amount = best_top10_retrieval - top10acc
            improvementt10 = f"improved T10 by {improvementt10_amount:.4f}" if epoch > 0 else "first save"

            # Delete previous best model if it exists
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"-- Deleted previous best model: {os.path.basename(best_model_path)}")

            # Update best loss and save new best model
            best_top10_retrieval = top10acc
            print(f"-- New best test loss! Saving 'improve' model... ({improvementt10})")
            best_model_path = save_checkpoint(model, optimizer, epoch+1, test_loss, 
                                    base_name="CVUSAGround2Satellite_improved")
            saved=True

        # ROUTINE SAVING + DETAILED METRICS (every 5 epochs)
        if (epoch + 1) % 5 == 0:           
            # Routine save
            print("   Routine save...")
            save_checkpoint(model, optimizer, epoch+1, test_loss, 
                                    base_name="CVUSAGround2Satellite_routine")
            
            
            print(f"   ✅ Epoch {epoch + 1} detailed evaluation complete")
    
    print(f"\n🎉 TRAINING COMPLETED!")
    print(f"🏆 Best test loss achieved: {best_test_loss:.4f}")
    print(f"📁 All models saved in: {os.getcwd()}")
    if best_model_path:
        print(f"🥇 Best model: {os.path.basename(best_model_path)}")

    if config.USE_WANDB:
        wandb.finish()
        
def main():
    [configpath, checkpointpath] = download_taming_vqgan(version=16, kaggle_flag=config.KAGGLE_FLAG)     
    model, _, device = build_model(configpath, checkpointpath, getDevice())

    train_loader, test_loader = CVUSADataset.create_dataloaders(
        data_root=config.DATA_ROOT,
        batch_size=config.BATCH_SIZE
    )

    train_model_with_evaluation(model, train_loader, test_loader, num_epochs=config.NUM_EPOCHS, lr=config.LEARNING_RATE)

if __name__ == "__main__":
    main()