import urllib.request
import os
import time

def generate_face_database(total_images=150):
    # Pastikan folder data tersedia di direktori root LOWCAM
    target_folder = 'data'
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
        print(f"[+] Folder '{target_folder}' berhasil dibuat.")

    print("="*50)
    print(f"   LOWCAM DATA GENERATOR - MENGUMPULKAN {total_images} FOTO")
    print("="*50)
    print("Sedang mengunduh sampel wajah manusia asli/AI untuk machine.py...")
    print("Tunggu sebentar, kecepatan bergantung pada koneksi internetmu.\n")

    # Header User-Agent agar tidak diblokir oleh server penyedia gambar
    opener = urllib.request.build_opener()
    opener.addheaders = [('User-agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')]
    urllib.request.install_opener(opener)

    count = 0
    while count < total_images:
        try:
            # Mengambil wajah dari generator AI yang selalu unik tiap request
            url = "https://thispersondoesnotexist.com"
            filename = f"face_{count + 1}.jpg"
            filepath = os.path.join(target_folder, filename)
            
            urllib.request.urlretrieve(url, filepath)
            
            count += 1
            # Progress bar sederhana
            progress = (count / total_images) * 100
            print(f"[{count}/{total_images}] Berhasil mengunduh: {filename} ({progress:.1f}%)")
            
            # Jeda 0.6 detik untuk stabilitas koneksi dan menghindari rate limit
            time.sleep(0.6) 
            
        except Exception as e:
            print(f"[!] Gagal pada gambar ke-{count + 1}: {e}")
            print("Mencoba kembali dalam 2 detik...")
            time.sleep(2)

    print("\n" + "="*50)
    print(f"BERHASIL! {count} foto wajah sudah tersimpan di folder 'data/'.")
    print("Sekarang kamu bisa menjalankan: python main.py")
    print("="*50)

if __name__ == "__main__":
    generate_face_database(150)