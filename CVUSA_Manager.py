import csv
import os
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import torchvision.transforms as transforms
import config

def get_standard_transform(size=256):
    """Standard image transform: resize, normalize to [-1, 1]"""
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

def load_csv_data(csv_path):
    """
    Simple utility to load CSV data back into memory (no pandas)
    Returns: (data_rows, headers)
    """
    with open(csv_path, 'r', newline='', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)  # First row is headers
        data_rows = list(reader)  # Rest are data
    
    return data_rows, headers

# CVUSA CSV Preprocessor and Manager ---------------------------------------------------------------

class CVUSAPreprocessor:
        """
        This class contains functions to check and fix the structure of the CVUSA CSV files.
        It uses only built-in libraries (csv, os) to read, analyze, and write CSV files without pandas.
        """
        @staticmethod
        def check_csv_structure(csv_path): 
            """
            Check the structure of the CSV file and print sample rows (no pandas)
            """
            print(f"📁 Checking CSV structure: {csv_path}")
            
            # Check if file exists
            if not os.path.exists(csv_path):
                print(f"❌ File not found: {csv_path}")
                return None, None
            
            # Read first few lines to understand structure
            with open(csv_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                rows = []
                for i, row in enumerate(reader):
                    rows.append(row)
                    if i >= 4:  # Read first 5 rows
                        break
            
            if not rows:
                print(f"❌ Empty CSV file")
                return None, None
            
            first_row = rows[0]
            second_row = rows[1] if len(rows) > 1 else None
            
            print(f"🔍 Raw file inspection:")
            print(f"  First row:  {','.join(first_row)}")
            if second_row:
                print(f"  Second row: {','.join(second_row)}")
            
            # Check if first row looks like data (contains .png) rather than headers
            has_proper_headers = not any('.png' in cell for cell in first_row)
            
            # Count total rows
            with open(csv_path, 'r', newline='', encoding='utf-8') as file:
                total_rows = sum(1 for _ in csv.reader(file))
            
            if has_proper_headers:
                print(f"✅ File has proper headers")
                data_rows = rows[1:]  # Skip header
                headers = first_row
                actual_data_count = total_rows - 1
            else:
                print(f"⚠️  File has NO proper headers - first row is data!")
                print(f"📝 Using custom column names...")
                data_rows = rows  # All rows are data
                headers = ['satellite_path', 'ground_path', 'duplicate_ground_path']
                actual_data_count = total_rows
            
            print(f"\n📊 CSV Info:")
            print(f"  - Total rows: {total_rows}")
            print(f"  - Data rows: {actual_data_count}")
            print(f"  - Columns: {len(headers)} -> {headers}")
            
            print(f"\n📋 First 5 data rows:")
            for i, row in enumerate(data_rows[:5]):
                print(f"  Row {i+1}: {row}")
            
            print(f"\n🔍 Sample paths analysis:")
            if data_rows:
                sample_row = data_rows[0]
                for i, (header, value) in enumerate(zip(headers, sample_row)):
                    print(f"  Column {i+1} ({header}): {value}")
            
            return rows, has_proper_headers

        @staticmethod
        def fix_csv_paths(input_csv_path, output_csv_path=None):
            """
            Fix the CSV file paths according to the requirements (no pandas):
            - Column 2: Remove 'input' from filename (streetview/input0026840.png → streetview/0026840.png)
            - Column 3: Change to polarmap/normal/input{ID}.png
            """
            
            # Check structure first (this only reads first 5 rows for inspection)
            result = CVUSAPreprocessor.check_csv_structure(input_csv_path)
            if result[0] is None:
                return None
            
            sample_rows, has_proper_headers = result
            
            # Now read ALL rows from the file
            print(f"🔄 Reading complete file: {input_csv_path}")
            with open(input_csv_path, 'r', newline='', encoding='utf-8') as file:
                reader = csv.reader(file)
                all_rows = list(reader)
            
            print(f"📊 Complete file loaded: {len(all_rows)} total rows")
            
            print(f"\n🔧 Applying transformations...")
            
            # Determine data start and headers using the complete dataset
            if has_proper_headers:
                headers = all_rows[0]
                data_rows = all_rows[1:]
                print(f"  - Using existing headers: {headers}")
            else:
                headers = ['satellite_path', 'ground_path', 'polarmap_path']
                data_rows = all_rows  # All rows are data
                print(f"  - Using custom headers: {headers}")
            
            # Transform the data
            fixed_rows = []
            
            print(f"  - Processing {len(data_rows)} data rows...")
            print(f"  - Fixing column 2: removing 'input' from filenames AND changing .png to .jpg")
            print(f"  - Fixing column 3: changing to polarmap/normal/input{{ID}}.png")
            
            for row_idx, row in enumerate(data_rows):
                if len(row) < 3:
                    print(f"⚠️  Row {row_idx + 1} has fewer than 3 columns: {row}")
                    fixed_rows.append(row)  # Keep as-is
                    continue
                
                # Extract original values
                satellite_path = row[0]  # Keep as-is
                ground_path = row[1]     # Remove 'input'
                third_path = row[2]      # Convert to polarmap
                
                # Transform column 2: Remove 'input' from filename AND change .png to .jpg
                fixed_ground_path = ground_path.replace('input', '').replace('.png', '.jpg')
                
                # Transform column 3: Change to polarmap/normal/input{ID}.png
                try:
                    # Extract filename from original path
                    original_filename = Path(third_path).name
                    fixed_third_path = f"polarmap/normal/{original_filename}"
                except:
                    fixed_third_path = third_path  # Keep original if parsing fails
                
                # Create fixed row
                fixed_row = [satellite_path, fixed_ground_path, fixed_third_path]
                
                # Add any additional columns if they exist
                if len(row) > 3:
                    fixed_row.extend(row[3:])
                
                fixed_rows.append(fixed_row)
            
            # Show before/after comparison
            if data_rows:
                print(f"\n📊 Transformation Results:")
                print(f"Original sample row:")
                sample_orig = data_rows[0]
                for i, val in enumerate(sample_orig[:3]):
                    print(f"  Col {i+1}: {val}")
                
                print(f"\nFixed sample row:")
                sample_fixed = fixed_rows[0]
                for i, val in enumerate(sample_fixed[:3]):
                    print(f"  Col {i+1}: {val}")
            
            # Determine output path
            if output_csv_path is None:
                input_path = Path(input_csv_path)
                output_csv_path = input_path.parent / f"{input_path.stem}_fixed{input_path.suffix}"
            
            # Write the fixed CSV
            with open(output_csv_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                
                # Write headers
                writer.writerow(headers)
                
                # Write data
                writer.writerows(fixed_rows)
            
            print(f"\n✅ Fixed CSV saved to: {output_csv_path}")
            print(f"📊 Output summary: {len(headers)} columns, {len(fixed_rows)} data rows")
            
            return fixed_rows, headers

        @staticmethod
        def process_train_test_csvs(train_csv_path, test_csv_path):
            """
            Process both training and test CSV files (no pandas)
            """
            print("="*60)
            print("🚀 Processing Training and Test CSV Files")
            print("="*60)
            
            # Process training CSV
            print("\n📚 TRAINING CSV:")
            train_result = CVUSAPreprocessor.fix_csv_paths(train_csv_path)
            
            print("\n" + "="*60)
            
            # Process test CSV
            print("\n🧪 TEST CSV:")
            test_result = CVUSAPreprocessor.fix_csv_paths(test_csv_path)
            
            print("\n" + "="*60)
            print("✅ Both CSV files processed successfully!")
            
            return train_result, test_result

        @staticmethod
        def cvusa_split_complete():
            # Get script directory
            script_dir = Path(__file__).parent
            
            # Local file paths - CVUSA_subset in same directory as script
            data_dir = script_dir / "cvusa-subset-csvfixed"
            train_csv = os.path.join(config.DATA_ROOT, "train-19zl_fixed.csv")
            test_csv  = os.path.join(config.DATA_ROOT, "val-19zl_fixed.csv")
            
            # Check if directory exists
            if not data_dir.exists():
                print(f"❌ Error: Directory not found: {data_dir}")
                print(f"   Please ensure 'CVUSA_subset' folder is in the same directory as this script")
                exit(1)
            
            # Check if CSV files exist
            if not train_csv.exists():
                print(f"❌ Error: Training CSV not found: {train_csv}")
                exit(1)
            
            if not test_csv.exists():
                print(f"❌ Error: Test CSV not found: {test_csv}")
                exit(1)
            
            # Check structure of both files
            print("Checking CSV structures...")
            CVUSAPreprocessor.check_csv_structure(train_csv)
            print("\n" + "="*40)
            CVUSAPreprocessor.check_csv_structure(test_csv)
            
            print("\n" + "="*60)
            
            # Process both files - saves to CVUSA_subset/
            train_result, test_result = CVUSAPreprocessor.process_train_test_csvs(train_csv, test_csv)
            
            # Print the final file locations
            print(f"\n📁 Fixed files saved to:")
            print(f"   Training: {data_dir / 'train-19zl_fixed.csv'}")
            print(f"   Test:     {data_dir / 'val-19zl_fixed.csv'}")
            
            # Example of loading the data back
            print(f"\n🔄 Example: Loading fixed training data...")
            train_data, train_headers = load_csv_data(data_dir / "train-19zl_fixed.csv")
            print(f"   Headers: {train_headers}")
            print(f"   Data rows: {len(train_data)}")
            print(f"   Sample: {train_data[0] if train_data else 'No data'}")

# --------------------------------------------------------------------------------------------------

class CVUSADataset(Dataset):
    def __init__(self, csv_path, data_root, size=256, polar=True, is_train=True):
        self.data_root = data_root
        self.size = size
        self.polar = polar
        
        # Load file pairs from CSV
        self.file_pairs = []
        
        print(f"📂 Loading dataset from: {csv_path}")
        
        with open(csv_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)
            for row in reader:
                if len(row) < 3: continue
                
                bingmap_path, ground_path, polarmap_path = row[0], row[1], row[2]
                satellite_relative_path = polarmap_path if self.polar else bingmap_path
                
                satellite_full_path = os.path.join(data_root, satellite_relative_path)
                ground_full_path = os.path.join(data_root, ground_path)
                
                if os.path.exists(satellite_full_path) and os.path.exists(ground_full_path):
                    self.file_pairs.append((satellite_full_path, ground_full_path))

        print(f"✅ Found {len(self.file_pairs)} valid image pairs.")
        
        # --- MODIFIED: Create separate transform pipelines ---
        
        # Pipeline for TARGET images (Satellite). ALWAYS without augmentation.
        self.satellite_transform = get_standard_transform(size)

        # Pipeline for INPUT images (Ground). Augmentation is applied ONLY if is_train=True.
        if is_train:
            self.ground_transform = transforms.Compose([
                transforms.Resize((size, size)),
                transforms.RandomApply([
                    transforms.RandomAffine(degrees=5, translate=(0.05, 0.05)),
                    transforms.RandomPerspective(distortion_scale=0.3, p=0.5)
                ], p=0.5),
                transforms.RandomApply([
                    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
                    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
                ], p=0.5),
                # --- End of Augmentations ---
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
            print(f"   -> Mode: TRAINING (applying augmentations to ground images)")
        else:
            # For the validation set, the ground transform is the same as the satellite one (no augmentations).
            self.ground_transform = self.satellite_transform
            print(f"   -> Mode: VALIDATION (no augmentations)")

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        satellite_path, ground_path = self.file_pairs[idx]
        
        try:
            # Load images
            satellite_img = Image.open(satellite_path).convert('RGB')
            ground_img = Image.open(ground_path).convert('RGB')
            
            # --- MODIFIED: Apply the correct transform to each image ---
            satellite_tensor = self.satellite_transform(satellite_img)
            ground_tensor = self.ground_transform(ground_img)
            
            return {
                "satellite": satellite_tensor,  # The clean, non-augmented target
                "ground": ground_tensor         # The potentially augmented input
            }
        
        except Exception as e:
            print(f"❌ Error loading images at index {idx}: {e}")
            dummy_tensor = torch.zeros(3, self.size, self.size)
            return {"satellite": dummy_tensor, "ground": dummy_tensor}

       
    @classmethod
    def create_dataloaders(cls, data_root, batch_size=8, polar=True,
                        train_csv=None, test_csv=None):
        """Factory method to create train/test dataloaders"""
        
        if train_csv is None:
            train_csv = os.path.join(data_root, "train-19zl_fixed.csv")
        if test_csv is None:
            test_csv = os.path.join(data_root, "val-19zl_fixed.csv")

        train_dataset = cls(train_csv, data_root, 256, polar, is_train=True)
        test_dataset  = cls(test_csv,  data_root, 256, polar, is_train=False)

        train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
        test_loader  = DataLoader(test_dataset,  batch_size, shuffle=False)

        return train_loader, test_loader