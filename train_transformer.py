def main():
    fix_torch_import_issue(kaggle_flag=False)  # Set to True if running in Kaggle environment
    [configpath, checkpointpath] = download_taming_vqgan(version=16, kaggle_flag=False)     # Set to True if running in Kaggle environment
    pretrained_vqgan = load_vqgan_model(configpath, checkpointpath)                         # Load the VQGAN model using the downloaded files
    fullsystem_config, device = create_config(configpath)                                   # Create the complete system config and choose device
    fix_inject_top_k_p_filtering()                                                          # Inject filtering function into transformers module to fix import
    return

if __name__ == "__main__":
    main()

