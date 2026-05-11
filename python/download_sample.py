import os
import requests

def download_samples(output_dir="data/sample_images"):
    os.makedirs(output_dir, exist_ok=True)
    
    # 5 Sample images for testing
    urls = [
        "https://picsum.photos/seed/1/800/600",
        "https://picsum.photos/seed/2/800/600",
        "https://picsum.photos/seed/3/800/600",
        "https://picsum.photos/seed/4/800/600",
        "https://picsum.photos/seed/5/800/600"
    ]
    
    print(f"Downloading {len(urls)} sample images to {output_dir}...")
    for i, url in enumerate(urls):
        filename = os.path.join(output_dir, f"sample_{i}.jpg")
        if not os.path.exists(filename):
            headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) RAMS-Project/1.0'}
            response = requests.get(url, stream=True, headers=headers)
            if response.status_code == 200:
                with open(filename, 'wb') as f:
                    for chunk in response.iter_content(1024):
                        f.write(chunk)
                print(f"Downloaded {filename}")
            else:
                print(f"Failed to download {url}")
        else:
            print(f"{filename} already exists.")
            
    print("Done!")

if __name__ == "__main__":
    download_samples()
