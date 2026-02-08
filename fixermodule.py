def fix_torch_import_issue(kaggle_flag=False):
    if kaggle_flag:
        utils_file = '/kaggle/working/taming-transformers/taming/data/utils.py'
    else:
        utils_file = 'taming-transformers/taming/data/utils.py'

    # Read the file
    with open(utils_file, 'r') as f:
        content = f.read()

    # Replace the problematic import
    content = content.replace(
        'from torch._six import string_classes',
        'string_classes = str'
    )

    # Write back the fixed file
    with open(utils_file, 'w') as f:
        f.write(content)

    print("Fixed torch._six import issue!")



