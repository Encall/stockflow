"""
Main ETL Pipeline for Stock Data
"""

import os
from dotenv import load_dotenv
from minio_handler import MinioHandler
from silver import process_silver_layer
from gold import create_all_gold_files


def main():
    """
    Main function to run the ETL pipeline.
    Downloads data from MinIO, processes it, and uploads the results.
    """
    load_dotenv()
    print("🚀 Starting ETL Pipeline")
    
    # Define the single source of truth for the local data directory path
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    LOCAL_DATA_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, '..', 'data'))

    print("=" * 60)
    print("🚀 Starting ETL Pipeline")
    print(f"Data directory: {LOCAL_DATA_DIR}")
    print("=" * 60)
    
    minio_handler = MinioHandler()

    # Step 1: Download data from MinIO
    print("\n📥 Step 1: Downloading data from MinIO...")
    file_names = minio_handler.download_data(local_data_dir=LOCAL_DATA_DIR)
    print(f"✅ Downloaded {len(file_names)} files")
    
    # Step 2: Process Silver Layer (Clean & Check Dates)
    print("\n🧹 Step 2: Processing Silver Layer...")
    silver_results = process_silver_layer(local_data_dir=LOCAL_DATA_DIR, file_list=file_names)
    silver_success_count = len([r for r in silver_results.values() if r["status"] == "success"])
    silver_failed_count = len(file_names) - silver_success_count
    print(f"✅ Silver Layer processed for {silver_success_count} files.")

    # Step 3: Upload Silver data to MinIO
    print("\n📤 Step 3: Uploading Silver data to MinIO...")
    silver_uploaded_files = minio_handler.upload_data(local_data_dir=LOCAL_DATA_DIR, layer="silver")
    print(f"✅ Uploaded {len(silver_uploaded_files)} files to MinIO (silver layer).")

    # Step 4: Create Gold Features
    print("\n✨ Step 4: Creating Gold Layer...")
    gold_results = create_all_gold_files(local_data_dir=LOCAL_DATA_DIR)
    gold_success_count = len([r for r in gold_results.values() if r["status"] == "success"])
    gold_failed_count = len([r for r in gold_results.values() if r["status"] == "failed" and r.get('error') is not None])
    print(f"✅ Gold Layer created for {gold_success_count} files.")

    # Step 5: Upload Gold data to MinIO
    print("\n📤 Step 5: Uploading Gold data to MinIO...")
    gold_uploaded_files = minio_handler.upload_data(local_data_dir=LOCAL_DATA_DIR, layer="gold")
    print(f"✅ Uploaded {len(gold_uploaded_files)} files to MinIO (gold layer).")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Pipeline Summary")
    print("=" * 60)
    print(f"📥 Downloaded: {len(file_names)} files")
    print(f"🧹 Silver Layer Processed: {silver_success_count} files")
    print(f"📤 Silver Layer Uploaded: {len(silver_uploaded_files)} files")
    print(f"✨ Gold Layer Created: {gold_success_count} files")
    print(f"📤 Gold Layer Uploaded: {len(gold_uploaded_files)} files")
    print("-" * 20)
    print(f"❌ Failed Silver Processing: {silver_failed_count} files")
    print(f"❌ Failed Gold Creation: {gold_failed_count} files")
    
    if silver_failed_count > 0:
        print("\nFailed files during Silver processing:")
        for filename, result in silver_results.items():
            if result["status"] == "failed":
                print(f"  ❌ {filename}: {result['error']}")

    if gold_failed_count > 0:
        print("\nFailed gold files:")
        for filename, result in gold_results.items():
            if result["status"] == "failed":
                print(f"  ❌ {filename}: {result['error']}")

    print("\n🎉 ETL Pipeline completed!")


if __name__ == "__main__":
    main()
