"""
Main ETL Pipeline for Stock Data
- Download data from MinIO
- Clean data and save to cleaned directory
- Upload cleaned data back to MinIO
"""

from download_data import download_data
from clean_data import clean_all_files
from upload_data import upload_data

def main():
    """Run the complete ETL pipeline"""
    print("=" * 60)
    print("🚀 Starting ETL Pipeline")
    print("=" * 60)
    
    # Step 1: Download data from MinIO
    print("\n📥 Step 1: Downloading data from MinIO...")
    file_names = download_data()
    print(f"✅ Downloaded {len(file_names)} files")
    
    # Step 2: Clean downloaded data
    print("\n🧹 Step 2: Cleaning data...")
    results = clean_all_files(file_list=file_names)
    
    success_count = len([r for r in results.values() if r["status"] == "success"])
    failed_count = len([r for r in results.values() if r["status"] == "failed"])
    print(f"✅ Successfully cleaned: {success_count} files")
    
    # Step 3: Upload cleaned data back to MinIO
    print("\n📤 Step 3: Uploading cleaned data to MinIO...")
    uploaded_files = upload_data()
    print(f"✅ Uploaded {len(uploaded_files)} files to MinIO")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Pipeline Summary")
    print("=" * 60)
    print(f"📥 Downloaded: {len(file_names)} files")
    print(f"🧹 Cleaned: {success_count} files")
    print(f"📤 Uploaded: {len(uploaded_files)} files")
    print(f"❌ Failed: {failed_count} files")
    
    if failed_count > 0:
        print("\nFailed files:")
        for filename, result in results.items():
            if result["status"] == "failed":
                print(f"  ❌ {filename}: {result['error']}")
    
    print("\n🎉 ETL Pipeline completed!")
    
    return {
        "downloaded": file_names,
        "cleaned": results,
        "uploaded": uploaded_files
    }

if __name__ == "__main__":
    main()
