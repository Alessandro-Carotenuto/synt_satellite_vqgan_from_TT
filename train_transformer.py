import torch
import os
from fixermodule import fix_torch_import_issue, fix_inject_top_k_p_filtering
import torch.nn.functional as F 
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.optim as optim 
import config 
from CVUSA_Manager import CVUSADataset
from taming_interface import download_taming_vqgan, save_checkpoint,manual_forward_pass, getDevice, build_model, get_optimizer


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
        if batch_idx % 50 == 0:
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
    correct_predictions = 0
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
            correct = (predictions == target).sum().item()    #HOW MANY TRUE PERFECT?
            
            # Accumulate metrics
            total_loss += loss.item()                          
            total_tokens += target.numel()
            correct_predictions += correct
            num_batches += 1
    
    # Calculate averages metrics and statistics
    avg_loss = total_loss / num_batches
    token_accuracy = correct_predictions / total_tokens
    
    model.train()   # Switch back to training mode
    
    return {
        'loss': avg_loss,
        'token_accuracy': token_accuracy,
        'total_tokens': total_tokens,
        'total_batches': num_batches
    }

def train_model_with_evaluation(model, train_dataloader, test_dataloader, num_epochs=50, lr=5e-4):
    """
    Modified training with train/test split and overfitting detection
    """
    # Setup training parameters with weight decay
    device = getDevice()
    model = model.to(device)
    model = torch.compile(model)  # Can give 10-20% speedup
    
    # Optimizer with weight decay for regularization
    optimizer = get_optimizer(model, lr, weight_decay=0.1) #weight decay 0.01 was not enough
    scaler = GradScaler()                                                                               #create the scaler
    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)                            #Cosine Annealing for LR Scheduling
    
    print(f"Starting training for {num_epochs} epochs...")
    print(f"Training set: {len(train_dataloader)} batches")
    print(f"Test set: {len(test_dataloader)} batches")
    print(f"Learning rate: {lr}, Weight decay: 0.01")
    print(f"Token masking: {config.TOKEN_MASKING_SCHEDULING_START}, to {config.TOKEN_MASKING_SCHEDULING_END}, in {num_epochs}")
    print(f"Training on device: {device}")
    
    # Tracking variables
    best_test_loss = float('inf')
    previous_gap = 0
    best_model_path = None  #Track the last saved best model to delete it
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"EPOCH {epoch + 1}/{num_epochs}")
        print(f"{'='*60}")

        current_token_masking=1.0-(epoch/num_epochs)*(config.TOKEN_MASKING_SCHEDULING_END-config.TOKEN_MASKING_SCHEDULING_START)
        
        # TRAINING PHASE
        print("Training phase...")
        train_loss = train_one_epoch(model, train_dataloader, optimizer, scaler, device, tmask_pkeep=current_token_masking) #LAUNCH TRAINING FOR THIS EPOCH
        
        # EVALUATION PHASE  
        print("Evaluation phase...")
        test_metrics = evaluate_model(model, test_dataloader, device)                               #EVALUATE FOR THIS EPOCH
        test_loss = test_metrics['loss']
        test_accuracy = test_metrics['token_accuracy']
        
        # UPDATE LEARNING RATE
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
        print(f"    Test Accuracy:  {test_accuracy:.3f} ({test_accuracy*100:.1f}%)")
        print(f"    Learning Rate:  {current_lr:.2e}")
        print(f"    Loss Gap (abs): {current_gap:.4f}")

        if gap_status:
            print(f"   {gap_status}")
        
        # BEST MODEL SAVING (when test loss improves)
        if test_loss < best_test_loss:
            # Calculate improvement
            improvement_amount = best_test_loss - test_loss
            improvement = f"improved by {improvement_amount:.4f}" if epoch > 0 else "first save"
            
            # Delete previous best model if it exists
            if best_model_path and os.path.exists(best_model_path):
                os.remove(best_model_path)
                print(f"-- Deleted previous best model: {os.path.basename(best_model_path)}")
            
            # Update best loss and save new best model
            best_test_loss = test_loss
            print(f"-- New best test loss! Saving 'improve' model... ({improvement})")
            best_model_path = save_checkpoint(model, optimizer, epoch+1, test_loss, 
                                    base_name="CVUSAGround2Satellite_improved")
        
        # ROUTINE SAVING + DETAILED METRICS (every 5 epochs)
        if (epoch + 1) % 5 == 0:
            print(f"\nEPOCH {epoch + 1} - DETAILED EVALUATION:")
            
            # Calculate and display perplexity
            perplexity = torch.exp(torch.tensor(test_loss))
            print(f"   Test Perplexity: {perplexity:.2f}")
            print(f"      (Model is as 'confused' as choosing randomly from ~{perplexity:.0f} options)")
            
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