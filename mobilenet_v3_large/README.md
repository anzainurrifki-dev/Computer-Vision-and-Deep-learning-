# MobileNetV3-Large — hasil dan analisis P2 Transfer Learning

Klasifikasi dua kelas: `box_merah` dan `box_cokelat`. README ini membahas
hasil aktual model ini serta perbandingannya dengan empat model lain.
Seluruh angka diambil dari CSV hasil training,
riwayat per epoch, dan checkpoint yang telah dievaluasi ulang.

## 1. Kondisi eksperimen

| Komponen | Konfigurasi |
|---|---|
| Dataset | 200 JPEG: 100 gambar per kelas |
| Split | 156 train, 36 validation, 8 excluded; sama untuk semua model |
| Input | RGB 224×224; normalisasi ImageNet |
| Training | 10 epoch per mode, batch 16, seed 42 |
| Optimizer / scheduler | Adam / CosineAnnealingLR |
| Perangkat sesi aktual | CUDA, NVIDIA GeForce RTX 4050 Laptop GPU |
| Fallback script | CPU jika CUDA tidak tersedia |

Train memakai RandomResizedCrop, HorizontalFlip, dan ColorJitter. Validation
memakai resize sisi pendek 256 lalu center crop 224 tanpa augmentasi acak.
Split bersifat kronologis dengan guard band dua sampel pada masing-masing
sisi batas. Script memeriksa path duplikat dan gambar identik lintas split/kelas.
Desain dan pemetaan keluaran slide 23: [DESIGN.md](DESIGN.md).

| Mode | Bobot awal | Bagian dilatih | Learning rate awal |
|---|---|---|---|
| feature | ImageNet | Classifier; backbone beku dan eval | 1e-3 |
| partial | ImageNet | dua modul akhir features dan classifier | Backbone 1e-4, classifier 1e-3 |
| scratch | Acak | Semua parameter | 1e-3 |

## 2. Hasil tiga mode

| Mode | Best val accuracy | Best epoch | Epoch pertama ≥90% | Training (s) | Parameter dilatih |
|---|---:|---:|---:|---:|---:|
| feature | 100.00% | 1 | 1 | 15.882 | 1.232.642 |
| partial | 100.00% | 1 | 1 | 16.020 | 2.185.522 |
| scratch | 50.00% | 1 | — | 18.277 | 4.204.594 |

Parameter total model: **4.204.594**.
Tanda — berarti ambang 90% belum tercapai dalam 10 epoch. Checkpoint menyimpan
epoch dengan validation accuracy tertinggi; jika seri, epoch pertama dipilih.

| Mode | Train accuracy epoch 10 | Val accuracy epoch 10 | Train loss epoch 10 | Val loss epoch 10 |
|---|---:|---:|---:|---:|
| feature | 100.00% | 100.00% | 0.0025 | 0.0014 |
| partial | 100.00% | 100.00% | 0.0019 | 8.46e-05 |
| scratch | 94.87% | 50.00% | 0.1066 | 0.6962 |

Train accuracy dihitung saat batch mengalami augmentasi dan update bobot.
Validation dihitung dalam mode eval. Angka keduanya tidak berasal dari
kondisi input dan evaluasi yang sama.

- [Grafik validation accuracy per epoch](results/accuracy_per_epoch.png)
- Kurva loss, accuracy, durasi, dan LR: [feature](results/feature_curves.png), [partial](results/partial_curves.png), [scratch](results/scratch_curves.png).
- [Tabel eksperimen lengkap](results/experiments.md) dan [CSV sumber](results/experiments.csv).

## 3. Analisis model ini

- Feature dan partial mencapai 100% sejak epoch 1 sampai epoch 10. Partial belum memberi peningkatan accuracy dibanding backbone yang dibekukan.
- Scratch tetap 50% pada seluruh epoch, sementara train accuracy akhir 94.87%. Checkpoint terbaik hanya memprediksi box_cokelat pada validation. Accuracy 50% sama dengan memilih satu kelas pada validation yang seimbang.
- Feature melatih 1.232.642 parameter dan partial 2.185.522. Waktu training tercatat 15.882 versus 16.020 detik; kedua nilai berdekatan dalam satu sesi pengukuran.
- Baseline untuk model ini: feature extraction. Menambah bagian backbone yang dilatih belum menghasilkan accuracy yang lebih tinggi, dan scratch belum membedakan kedua kelas pada validation.

## 4. Analisis latency

| Mode | Median (ms) | Mean (ms) | P95 (ms) |
|---|---:|---:|---:|
| feature | 2.118 | 2.175 | 2.471 |
| partial | 2.290 | 2.285 | 2.456 |
| scratch | 2.157 | 2.165 | 2.231 |

Pengukuran: GPU yang sama, PyTorch 2.14.0+cu130, input 224×224, batch 1, CPU threads 1, 20 warm-up dan 100 iterasi.
Batang grafik menunjukkan median; penanda hitam menunjukkan P95. Data dipilih
dari kelompok dengan perangkat, pengaturan, dan hash gambar yang sama.
CSV sumber menyimpan riwayat pengukuran; tabel di atas memakai pengukuran GPU
terbaru yang juga dipakai pada grafik.

Latency mencakup forward pass, belum mencakup kamera, preprocessing,
postprocessing, dan ROS2. Acuan PPT adalah anggaran inferensi 35 ms dari
sekitar 67 ms/frame (15 FPS). Nilai pada laptop ini belum membuktikan FPS
keseluruhan pipeline atau kinerja pada komputer robot.

- [Grafik latency tiga mode](results/latency_bar.png) · [SVG untuk PPT](results/latency_bar.svg)
- [CSV latency sumber](results/latency.csv) · [Tabel pengukuran terpilih](results/latency_summary.md)

## 5. Perbandingan antar model

Tabel memakai mode **feature** untuk membandingkan waktu dan latency dengan
strategi training yang sama. Accuracy setiap mode tetap ditampilkan.

| Model | Feature accuracy | Partial accuracy | Scratch accuracy | Total parameter (juta) | Training feature (s) | Median feature (ms) |
|---|---:|---:|---:|---:|---:|---:|
| MobileNetV3-Small | 100.00% | 100.00% | 50.00% | 1.520 | 16.595 | 1.820 |
| **MobileNetV3-Large** | 100.00% | 100.00% | 50.00% | 4.205 | 15.882 | 2.118 |
| EfficientNet-B0 | 100.00% | 100.00% | 100.00% | 4.010 | 16.952 | 2.805 |
| ResNet-18 | 100.00% | 100.00% | 100.00% | 11.178 | 16.406 | 1.617 |
| ResNet-50 | 100.00% | 100.00% | 100.00% | 23.512 | 19.462 | 2.852 |

Dibanding MobileNetV3-Small, parameter total lebih besar (4,205 versus 1,520 juta), tetapi feature/partial memiliki accuracy sama dan scratch sama-sama 50%. Median feature lebih tinggi (2,118 versus 1,820 ms). Eksperimen ini belum menunjukkan manfaat memilih varian Large untuk dua kelas tersebut.

### Kesimpulan perbandingan

- Semua model dengan feature/partial mencapai best validation accuracy 100%. Accuracy ini belum cukup untuk menentukan satu pemenang.
- **Ukuran model:** MobileNetV3-Small feature merupakan kandidat paling kecil: 1,520 juta parameter dan checkpoint sekitar 5,93 MiB.
- **Latency pada mode feature:** ResNet-18 tercatat paling cepat, median 1,617 ms. Di seluruh 15 konfigurasi, ResNet-18 partial mencatat median paling rendah, 1,396 ms.
- **Scratch:** kedua MobileNet tertahan di 50%; EfficientNet-B0 membutuhkan 8 epoch untuk 100%, ResNet-50 2 epoch, dan ResNet-18 1 epoch. Manfaat transfer learning paling terlihat pada MobileNet dan EfficientNet dalam eksperimen ini.
- **Pemilihan awal:** MobileNetV3-Small feature untuk prioritas ukuran checkpoint; ResNet-18 feature untuk baseline dengan latency GPU rendah. Ukur ulang pada perangkat robot dan data lintas sesi sebelum keputusan deployment.

- [Grafik accuracy lima model](results/comparison/accuracy_comparison.png)
- [Grafik latency 15 konfigurasi](results/comparison/latency_comparison.png) · [SVG](results/comparison/latency_comparison.svg)
- [Tabel training 15 konfigurasi](results/comparison/experiments.md) · [Tabel latency](results/comparison/latency.md)

## 6. Batas evaluasi dan tindak lanjut

Setiap kelas berasal dari satu video. Frame validation berbeda dari train,
tetapi masih berasal dari sesi pengambilan yang sama. Resolusi dan latar
antar kelas juga berbeda. Guard band dan pemeriksaan duplikat mengurangi
risiko frame bocor, tetapi belum menghilangkan kemiripan antarsampel dan
petunjuk dari kondisi pengambilan.

Validation hanya 18 gambar per kelas: satu kesalahan mengubah accuracy total
sekitar 2,78 poin persentase. Hasil 100% berarti 36/36 benar pada holdout ini,
bukan jaminan generalisasi. Belum ada test set independen atau eksperimen
dengan beberapa seed.

Tindak lanjut: rekam tiap kelas pada beberapa sesi, cahaya, jarak, orientasi,
dan latar; pisahkan sesi train/validation/test; ulangi eksperimen dan latency
untuk melihat kestabilan hasil. Penyebab scratch MobileNet tertahan di 50%
memerlukan eksperimen terkontrol tambahan sebelum disimpulkan.

## 7. Menjalankan dan memeriksa proyek

Dari folder `mobilenet_v3_large`:

```bash
python3 -m pip install -r requirements.txt
python3 split.py
python3 train.py
python3 latency.py --mode all --device auto --threads 1 --image dataset_raw/box_merah/box_merah__frame_000000000.jpg
python3 compare.py
```

Training default menjalankan ketiga mode. Gunakan `--mode feature`,
`--mode partial`, atau `--mode scratch` untuk satu mode. Menjalankan training
akan memperbarui checkpoint dan hasil mode tersebut. Bobot ImageNet perlu
cache atau akses jaringan untuk feature/partial.

Menggambar ulang dari hasil yang sudah tersedia:

```bash
python3 train.py --plot-only
python3 plot_latency.py --device cuda
python3 plot_latency.py --comparison --device cuda
```

`--device cuda` pada plotter memilih data GPU di CSV dan tidak membutuhkan
akses ke GPU. `compare.py` memakai hasil lima folder saudara yang tersedia;
proyek tetap dapat dipakai sendiri. Perbandingan antarmodel pada README ini
merangkum hasil eksperimen yang tersedia dan perlu ditinjau setelah training baru.

```bash
python3 -m unittest discover -s tests -v
```

Tiap proyek memiliki 7 tes: split, duplikat lintas split, freezing dan optimizer,
training/checkpoint/latency pada data sintetis sementara, serta seleksi data
grafik yang mencegah pencampuran perangkat/pengaturan dan memeriksa timestamp
serta nilai tidak valid. Artefak sintetis tidak masuk tabel hasil aktual.

Kelima belas checkpoint telah dievaluasi ulang pada CPU; accuracy cocok dengan
laporan training. Confusion matrix dan hasil pemeriksaan tersedia dalam
[verification.json](results/verification.json).
