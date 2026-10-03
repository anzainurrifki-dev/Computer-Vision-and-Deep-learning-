# Hasil eksperimen aktual

| Arsitektur | Mode | Best val accuracy | Best epoch | Epoch ≥90% | Waktu training (s) | Parameter dilatih / total | Epoch dijalankan |
|---|---|---:|---:|---:|---:|---:|---:|
| mobilenet_v3_large | feature | 1.0000 | 1 | 1 | 15.882 | 1232642 / 4204594 | 10 |
| mobilenet_v3_large | partial | 1.0000 | 1 | 1 | 16.020 | 2185522 / 4204594 | 10 |
| mobilenet_v3_large | scratch | 0.5000 | 1 | — | 18.277 | 4204594 / 4204594 | 10 |

Hanya konfigurasi yang memiliki hasil training dicantumkan. Tanda — berarti data tidak tersedia atau ambang belum tercapai.
CSV lama tetap dapat dibaca; jumlah parameter dan epoch yang belum dicatat tidak diisi dengan perkiraan.
Validation berasal dari holdout temporal satu video per kelas, sehingga bukan evaluasi lintas sesi.
