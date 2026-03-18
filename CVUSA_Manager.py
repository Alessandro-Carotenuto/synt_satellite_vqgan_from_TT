import csv
import os
from pathlib import Path    
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as transforms
import config
import numpy as np
import random


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


        
        @staticmethod
        def polarize(image, output_size=None):
            """
            Converts a top-down aerial image into a polar projection.
            Think of it as unrolling the aerial image from the center outward:
            the center becomes the top row, the edges become the bottom row.
            """
            from scipy.ndimage import map_coordinates

            img = np.array(image.convert('RGB')).astype(np.float32)
            Ds  = img.shape[0]  # input is square

            Hv, Wv = output_size if output_size else (Ds, Ds)

            # For each pixel in the output, figure out where to sample in the input.
            # Rows control radius (how far from center), columns control angle (0 to 2pi).
            row, col = np.meshgrid(np.arange(Hv), np.arange(Wv), indexing='ij')

            radius = (Ds / 2.0) * (Hv - row) / Hv
            angle  = (2.0 * np.pi / Wv) * col

            src_row = Ds / 2.0 - radius * np.cos(angle)
            src_col = Ds / 2.0 + radius * np.sin(angle)

            channels = [
                map_coordinates(img[:, :, c], [src_row, src_col], order=1, mode='nearest')
                for c in range(3)
            ]

            return Image.fromarray(np.stack(channels, axis=-1).clip(0, 255).astype(np.uint8))


        @staticmethod
        def apply_polar_to_dataset(csv_path, data_root, output_size=None, num_workers=4):
            """
            One-time preprocessing: reads aerial images from a CSV,
            applies the polar transform, and saves results to polarmap/normal/.
            Already processed images are skipped automatically.
            """
            from concurrent.futures import ThreadPoolExecutor, as_completed

            # Read all unique aerial paths from column 1
            with open(csv_path, 'r', newline='', encoding='utf-8') as f:
                reader = csv.reader(f)
                next(reader)
                aerial_paths = list(dict.fromkeys(row[0] for row in reader if row))

            print(f"Found {len(aerial_paths)} aerial images to process")

            def process_one(rel_path):
                src = os.path.join(data_root, rel_path)
                dst = os.path.join(data_root, "polarmap", "normal", Path(rel_path).name)

                if os.path.exists(dst):
                    return 'skipped'

                try:
                    os.makedirs(os.path.dirname(dst), exist_ok=True)
                    CVUSAPreprocessor.polarize(Image.open(src), output_size).save(dst)
                    return 'ok'
                except Exception as e:
                    print(f"  Failed on {src}: {e}")
                    return 'failed'

            counts = {'ok': 0, 'skipped': 0, 'failed': 0}
            total  = len(aerial_paths)

            with ThreadPoolExecutor(max_workers=num_workers) as pool:
                futures = {pool.submit(process_one, p): p for p in aerial_paths}

                for i, future in enumerate(as_completed(futures), start=1):
                    counts[future.result()] += 1

                    if i % 500 == 0 or i == total:
                        print(f"  {i}/{total} - saved {counts['ok']}, "
                            f"skipped {counts['skipped']}, failed {counts['failed']}")

            print(f"Done. {counts['ok']} generated, "
                f"{counts['skipped']} already existed, {counts['failed']} errors.")
# --------------------------------------------------------------------------------------------------

class CVUSADataset(Dataset):
    def __init__(self, csv_path, data_root, size=256, is_train=True):
        self.data_root = data_root
        self.size = size
        self.is_train = is_train
        self.file_pairs = []

        print(f"📂 Loading dataset from: {csv_path}")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV not found at {csv_path}")

        with open(csv_path, 'r', newline='', encoding='utf-8') as file:
            # We use a standard reader to handle various header names
            reader = csv.reader(file)
            header = next(reader)
            
            # Map columns based on your new dataset structure
            # Adjust these indices if your CSV columns are in a different order
            # Usually: Col 0 = Satellite/Polar, Col 1 = Ground
            for row in reader:
                if len(row) < 2: continue
                
                # Standardizing paths for Windows/Linux compatibility
                polar_rel = row[0].replace('\\', os.sep).strip()
                ground_rel = row[1].replace('\\', os.sep).strip()
                
                polar_path = os.path.join(data_root, polar_rel)
                ground_path = os.path.join(data_root, ground_rel)

                if os.path.exists(polar_path) and os.path.exists(ground_path):
                    self.file_pairs.append((polar_path, ground_path))

        print(f"✅ Found {len(self.file_pairs)} valid image pairs.")

        self.base_transform = get_standard_transform(size)

    def __len__(self):
        return len(self.file_pairs)

    def __getitem__(self, idx):
        polar_path, ground_path = self.file_pairs[idx]
        
        # Load images
        polar_img = Image.open(polar_path).convert('RGB')
        ground_img = Image.open(ground_path).convert('RGB')

        # Synchronized Augmentation for Training
        if self.is_train:
            # Horizontal Flip (Geometric consistency: flip both)
            if random.random() > 0.5:
                polar_img = TF.hflip(polar_img)
                ground_img = TF.hflip(ground_img)

            # Color jitter only on ground to make model robust
            if random.random() > 0.2:
                ground_img = transforms.ColorJitter(0.1, 0.1, 0.1)(ground_img)

        # Apply final transforms
        polar_tensor = self.base_transform(polar_img)
        ground_tensor = self.base_transform(ground_img)

        # In your transformer logic: 
        # 'satellite' = target (Polar)
        # 'ground' = condition (Streetview)
        return {"satellite": polar_tensor, "ground": ground_tensor}

    @classmethod
    def create_dataloaders(cls, data_root, batch_size=8, train_csv=None, test_csv=None):
        # Default filenames if not provided
        train_csv = train_csv or os.path.join(data_root, "train.csv")
        test_csv = test_csv or os.path.join(data_root, "val.csv")

        train_dataset = cls(train_csv, data_root, size=256, is_train=True)
        test_dataset  = cls(test_csv,  data_root, size=256, is_train=False)

        # Set num_workers=0 if you want to avoid the Windows "spawn" prints entirely
        # but 4 is usually fine with the fix I provided above.
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False, num_workers=4)

        return train_loader, test_loader