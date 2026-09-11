# CNN Convolution Accelerator (RTL)

Một bộ tăng tốc phần cứng (hardware accelerator) cho phép tính tích chập (convolution) trong mạng CNN, thiết kế bằng RTL (Verilog/SystemVerilog), hướng đến ứng dụng Edge AI. Engine được tham số hóa (parameterizable) để có thể tái sử dụng cho nhiều layer / nhiều mô hình CNN khác nhau, thay vì cố định cho một kiến trúc mạng cụ thể.

## Mục tiêu dự án

Thiết kế một **convolution engine 3×3** làm lõi tính toán trung tâm của một CNN accelerator, có khả năng:

- Xử lý phép tích chập 3×3 với số input/output channel tùy chỉnh (tham số hóa runtime, không hard-code lúc tổng hợp)
- Hỗ trợ **tiling**: xử lý feature map lớn hơn dung lượng buffer on-chip, có xử lý đúng vùng chồng lấp (halo) giữa các tile
- Hỗ trợ **weight streaming**: nạp trọng số dần từ bộ nhớ ngoài khi weight của layer vượt quá buffer on-chip, kèm double buffering để giảm thời gian chờ
- Tích hợp ReLU và quantization (INT8) ngay trong pipeline tính toán

Đây là dự án cá nhân, thực hiện với mục tiêu học tập và làm portfolio kỹ thuật RTL design & verification, không nhằm mục đích thương mại hóa hay tape-out thực tế.

## Bài toán & Workload tham chiếu

Engine được kiểm thử và đánh giá hiệu năng bằng một mạng CNN classification nhỏ trên CIFAR-10:

```
Input: 32x32x3 (CIFAR-10)
Conv1: 3x3, 3→16 channels, stride 1, pad 1  → ReLU → MaxPool 2x2
Conv2: 3x3, 16→32 channels, stride 1, pad 1 → ReLU → MaxPool 2x2
Conv3: 3x3, 32→64 channels, stride 1, pad 1 → ReLU → MaxPool 2x2
FC1: 64*4*4 → 128 → ReLU
FC2: 128 → 10 (classify)
```

Model được train bằng PyTorch, sau đó quantize về INT8 để phù hợp với engine phần cứng.

## Kiến trúc tổng quan

Engine xử lý tuần tự theo từng output channel, cộng dồn kết quả qua các input channel. Các khối chính:

| Khối | Chức năng |
|---|---|
| Line Buffer & Window Generator | Giữ 3 hàng feature map, trích cửa sổ 3×3 mỗi chu kỳ, xử lý padding biên |
| PE Array (9 PE) | 9 phép nhân-cộng INT8 song song cho một cửa sổ 3×3 |
| Channel Accumulator | Cộng dồn partial sum qua input channel, hỗ trợ read-modify-write khi cần tiling |
| ReLU | Cắt giá trị âm |
| Quantizer | Scale + shift kết quả 32-bit về lại INT8, tham số theo từng layer |
| Tiling Controller | Chia feature map lớn thành tile, quản lý vùng overlap (halo) |
| Weight Buffer & Streaming | Nạp weight on-chip, double buffering khi weight lớn hơn buffer |
| Top-level FSM | Điều phối toàn bộ pipeline, đọc tham số runtime (C_in, C_out, H, W, tile size...) |

*(Sơ đồ khối chi tiết sẽ được cập nhật trong thư mục `docs/`.)*

## Công cụ & Flow

- **RTL**: SystemVerilog
- **Verification**: Testbench SystemVerilog, so khớp kết quả từng layer với golden model tham chiếu
- **Golden model**: Python (PyTorch + NumPy), mô phỏng bit-accurate phép tính INT8
- **Tổng hợp mã nguồn mở**: Yosys, đánh giá area/timing với OpenROAD + SKY130 PDK

## Trạng thái dự án

- [ ] Golden model (Python) cho toàn bộ pipeline conv + ReLU + quantize
- [ ] RTL: PE Array
- [ ] RTL: Channel Accumulator
- [ ] RTL: Line Buffer / Window Generator (có padding)
- [ ] RTL: Quantizer
- [ ] RTL: Top-level Controller FSM
- [ ] RTL: Tiling & Weight Streaming
- [ ] Verification: testbench so khớp golden model
- [ ] Tổng hợp thử với Yosys/OpenROAD (SKY130)
- [ ] Đánh giá hiệu năng: throughput, latency, ước lượng power/area

## Cấu trúc thư mục (dự kiến)

```
.
├── rtl/            # Mã nguồn SystemVerilog
├── tb/             # Testbench
├── model/          # Golden model Python, script train/quantize
├── sim/            # Script mô phỏng, kịch bản test
├── synth/          # Script tổng hợp Yosys/OpenROAD
├── docs/           # Sơ đồ khối, ghi chú thiết kế
└── README.md
```

## Ghi chú

Dự án đang trong quá trình phát triển. README sẽ được cập nhật khi có kết quả cụ thể (waveform, số liệu tổng hợp, so sánh hiệu năng).

## Tác giả

Sinh viên ngành [Kỹ thuật máy tính/Điện tử], hướng chuyên môn RTL Design & Verification.
