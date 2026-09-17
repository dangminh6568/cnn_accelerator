# CNN Accelerator IP Core (RTL)

IP core phần cứng tăng tốc tính toán cho mạng CNN (Conv + Fully-Connected), thiết kế bằng SystemVerilog, hướng đến Edge AI/NPU cỡ nhỏ. Lõi tính toán dùng chung 1 mảng MAC cho cả lớp Conv lẫn FC, giao tiếp qua chuẩn AXI4.

## Mục tiêu

Thiết kế một **MAC Array song song hóa theo channel, weight-stationary**, dùng chung cho cả Convolution và Fully-Connected — không cần 2 khối phần cứng riêng biệt:

- Xử lý Conv 3×3 (và mở rộng được cho kernel size khác) với C_in/C_out tùy chỉnh runtime
- Xử lý FC layer bằng chính mảng MAC đó (FC = trường hợp đặc biệt của conv với kernel size 1)
- Tận dụng nguyên lý weight-stationary để giảm năng lượng đọc SRAM (đọc SRAM tốn năng lượng gấp nhiều lần so với 1 phép nhân 8-bit)
- Giao tiếp chuẩn hóa qua AXI4-Lite (config) và AXI4-Stream (data/weight/output)

Dự án cá nhân, mục tiêu học tập và portfolio kỹ thuật RTL design & verification.

## Bài toán & Workload tham chiếu

```
Input: 32x32x3 (CIFAR-10)
Conv1: 3x3, 3→16 channels, stride 1, pad 1  → ReLU → MaxPool 2x2
Conv2: 3x3, 16→32 channels, stride 1, pad 1 → ReLU → MaxPool 2x2
Conv3: 3x3, 32→64 channels, stride 1, pad 1 → ReLU → MaxPool 2x2
FC1: 64*4*4 → 128 → ReLU
FC2: 128 → 10 (classify)
```

Model train bằng PyTorch, quantize về INT8 để khớp với engine phần cứng.

## Quá trình đưa ra quyết định kiến trúc

Dự án đã cân nhắc qua 3 hướng trước khi chốt phương án cuối:

| Tiêu chí | 9-PE Spatial Convolver (thử đầu tiên) | Systolic Array 2D thuần / GEMM qua im2col (thử thứ hai) | **Channel-Parallel Weight-Stationary MAC Array (chốt)** |
|---|---|---|---|
| Khả năng chạy Conv 3×3 | Tốt | Rất tốt (qua im2col) | Rất tốt |
| Khả năng chạy FC | **Liệt hoàn toàn** — gắn chết với cửa sổ 3×3 | Chạy được nhưng cần thêm biến đổi | Tận dụng 100% cùng phần cứng, không cần khối riêng |
| Độ phức tạp FSM/control | Trung bình | Rất phức tạp (skewing, address generation) | Vừa phải, ánh xạ trực quan |
| Hiệu suất sử dụng PE | 100% cho đúng 3×3, kém cho kernel khác | Thấp nếu C_in/C_out lẻ | Cao, linh hoạt theo kernel size |
| Tối ưu năng lượng | Tốn công đọc line-buffer liên tục | Tối ưu truyền nội bộ nhưng phức tạp | Tối ưu nhờ weight-stationary (giảm truy xuất SRAM) |
| Phù hợp quy mô IP core gọn (dễ timing closure trên SKY130) | Được | Quá cồng kềnh cho Edge AI | Phù hợp nhất |

**Lý do chốt phương án 3**: đây là kiến trúc duy nhất giải quyết được đồng thời cả Conv và FC bằng chung 1 phần cứng, không phải trả giá về data duplication (im2col) hay độ phức tạp timing (skewing) như phương án Systolic-GEMM, đồng thời quy mô đủ gọn để khả thi tổng hợp/timing closure trong phạm vi đồ án. Chi tiết 2 phương án đã thử (bao gồm RTL của Systolic-GEMM + skew) được giữ lại tại `docs/explored/` làm tài liệu so sánh.

## Kiến trúc

### Nguyên lý dataflow: Weight-Stationary, song song hóa theo Channel

```
                    Input Feature Stream (broadcast / shifted)
                         |        |        |        |
                         v        v        v        v
              +--------------------------------------+
Weight Reg -- | PE(0,0)   PE(0,1)   ...    PE(0,N)    | ---> pSum Accumulator (output channel 0)
              +--------------------------------------+
Weight Reg -- | PE(1,0)   PE(1,1)   ...    PE(1,N)    | ---> pSum Accumulator (output channel 1)
              +--------------------------------------+
              |   ...        ...            ...      |
              +--------------------------------------+
Weight Reg -- | PE(M,0)   PE(M,1)   ...    PE(M,N)    | ---> pSum Accumulator (output channel M)
              +--------------------------------------+
```

- **Chiều M (hàng)**: song song hóa theo **output channel** (`C_out`)
- **Chiều N (cột)**: song song hóa theo **input channel** (`C_in`) hoặc spatial pixel, tùy giai đoạn
- **Weight-stationary**: mỗi PE giữ cố định 1 giá trị weight trong thanh ghi nội bộ suốt 1 lượt tính; input (activation) được truyền quét qua mảng để nhân với các weight này — 1 giá trị input được tái sử dụng đồng thời cho nhiều output channel (giảm mạnh số lần đọc SRAM)
- **Kernel 3×3 xử lý bằng cách lặp tuần tự 9 tap**: với mỗi vị trí kernel (kx, ky) trong 9 tap, nạp bộ weight tương ứng, chạy 1 lượt qua mảng, cộng dồn kết quả vào accumulator — sau 9 lượt, output pixel hoàn tất. Với FC (kernel size = 1), chỉ cần 1 lượt duy nhất, dùng chung mảng và accumulator này.
- Nếu `C_in`/`C_out` của layer lớn hơn kích thước vật lý M×N của mảng, chạy nhiều lượt tuần tự (time-multiplexing), cộng dồn thêm vào accumulator.

### Kích thước mảng đề xuất

Mảng MAC **4×4 hoặc 4×8** (INT8) — quy mô đủ nhỏ để khả thi đóng timing ~150–200 MHz trên SKY130 qua OpenROAD, không thiếu hụt diện tích khi route, phù hợp phạm vi đồ án hơn so với mảng lớn 16×16/32×32 kiểu TPU.

### Sơ đồ khối tổng thể

```
Input Buffer (SRAM 2-port, layout NHWC/CHW) ──┐
                                                ▼
Line Buffer & Window Generator (Conv) ─── MAC Array (M×N, weight-stationary)
   (bỏ qua/bypass khi chạy FC)                 │
                                                ▼
Weight Buffer (Ping-Pong SRAM) ──────►    pSum Accumulator
                                                │
                                                ▼
                                         ReLU → Quantizer
                                                │
                                                ▼
                                         Output Buffer / AXI4-Stream
```

| Khối | Chức năng |
|---|---|
| Input Buffer (SRAM 2-port) | Lưu feature map/activation theo layout channel (NHWC hoặc CHW tùy bus width) |
| Line Buffer & Window Generator | Sinh cửa sổ 3×3×C_in mỗi bước cho Conv; bypass khi chạy FC (input đọc thẳng dạng vector) |
| MAC Array (M×N, weight-stationary) | Lõi tính toán dùng chung cho Conv và FC, lặp tuần tự theo tap kernel |
| Weight Buffer (Ping-Pong) | Chứa block weight INT8, double buffering để nạp trước weight lượt kế tiếp |
| pSum Accumulator | Cộng dồn qua các tap kernel và các lượt time-multiplexing channel |
| ReLU | Cắt giá trị âm |
| Quantizer | Scale + shift về lại INT8, tham số theo từng layer |
| Tiling Controller | Chia feature map lớn thành tile, quản lý overlap (halo) — *giai đoạn nâng cao* |
| Top-level FSM | Điều phối toàn bộ: load config, load weight, chạy tap/channel loop, đồng bộ AXI4 |
| AXI4-Lite Config Interface | Thanh ghi cấu hình: `K_size`, `Stride`, `C_in`, `C_out`, `Scale`, `Shift`, `Pad_en` |
| AXI4-Stream Data Interface | Nạp input/weight, xuất output dạng stream |

## Đặc tả chức năng phần cứng

### 1. MAC Array (M×N PE)
- Mỗi PE: 1 multiplier INT8×INT8 + 1 adder, giữ 1 giá trị weight cố định trong thanh ghi nội bộ suốt 1 lượt tính (nạp qua Weight Loading Path trước khi compute)
- Input broadcast/shift tới các PE theo hàng hoặc cột tùy ánh xạ (không cần skew phức tạp như thiết kế systolic-GEMM)
- Với Conv: lặp 9 lần (từng tap kernel 3×3), mỗi lần dùng 1 bộ weight khác nạp vào PE, cộng dồn vào pSum Accumulator
- Với FC: chạy 1 lượt duy nhất, kernel size = 1, dùng chung accumulator

### 2. Weight Buffer (Ping-Pong)
- 2 buffer luân phiên: buffer A đang cấp weight cho MAC Array tính, buffer B nạp trước weight của tap/lượt kế tiếp từ AXI4-Stream
- Giảm thời gian chờ giữa các lượt tap/channel

### 3. Line Buffer & Window Generator (chỉ dùng cho Conv)
- Giữ 3 hàng feature map, sinh cửa sổ 3×3×C_in mỗi bước
- Xử lý padding biên theo `Pad_en`
- Bypass khối này khi FSM ở chế độ FC (input đọc thẳng từ Input Buffer dạng vector, không cần cửa sổ trượt)

### 4. pSum Accumulator
- Cộng dồn kết quả qua 9 tap kernel (Conv) hoặc qua các lượt time-multiplexing khi C_in/C_out > kích thước vật lý mảng
- Cộng bias ở lượt cuối cùng trước khi chuyển ReLU
- Reset khi bắt đầu 1 output pixel/output neuron mới

### 5. ReLU
- So sánh với 0, giữ nguyên nếu dương

### 6. Quantizer
- Nhân scale + dịch bit, clamp về INT8
- Đọc `Scale`, `Shift` riêng theo từng layer từ AXI4-Lite config

### 7. Tiling Controller *(giai đoạn nâng cao)*
- Chia feature map lớn thành tile vừa Input Buffer
- Quản lý vùng overlap (halo) giữa các tile
- Khuyến nghị tách module độc lập, giao tiếp qua địa chỉ/offset

### 8. Top-level Controller FSM
- State tối thiểu: `IDLE` → `LOAD_CONFIG` → `LOAD_WEIGHT` → `COMPUTE_TAP` (lặp 9 lần nếu Conv, 1 lần nếu FC) → `COMPUTE_CHANNEL_GROUP` (nếu cần time-multiplexing) → `DONE`
- Biết chế độ đang chạy (Conv hay FC) để bật/tắt Line Buffer, số vòng lặp tap tương ứng

### 9. AXI4-Lite Config Interface
- Thanh ghi: `K_size` (kernel size, hỗ trợ 1/3/5), `Stride`, `C_in`, `C_out`, `Scale`, `Shift`, `Pad_en`, `H`, `W`
- Cấu hình trước khi kích hoạt `start` qua AXI4-Lite write

### 10. AXI4-Stream Data Interface
- 1 stream cho input activation, 1 stream cho weight, 1 stream cho output — mỗi stream có `TVALID/TREADY/TLAST/TDATA` theo chuẩn AXI4-Stream

## Lộ trình triển khai

1. Golden model Python (numpy, bit-accurate INT8), mô phỏng cả đường Conv và đường FC dùng chung công thức MAC lặp tap
2. RTL: 1 PE đơn (weight-stationary, giữ weight trong thanh ghi), verify phép nhân-cộng cơ bản
3. RTL: MAC Array nhỏ (ví dụ 2×2), verify cơ chế song song hóa channel + accumulator qua vài tap giả lập
4. Mở rộng MAC Array lên kích thước đầy đủ (4×4 hoặc 4×8)
5. Weight Buffer (bắt đầu single buffer, thêm ping-pong sau)
6. Line Buffer + Window Generator cho đường Conv, xác nhận bypass đúng khi chuyển sang FC
7. ReLU + Quantizer
8. Top-level FSM: chạy tự động qua tap loop (Conv) và single-pass (FC), qua toàn bộ layer của model CIFAR-10
9. AXI4-Lite Config + AXI4-Stream Data Interface
10. *(Nâng cao, tùy thời gian)* Tiling Controller cho feature map lớn hơn Input Buffer

## Công cụ & Flow

- **RTL**: SystemVerilog
- **Verification**: Testbench SystemVerilog, so khớp từng layer (cả Conv và FC) với golden model
- **Golden model**: Python (PyTorch + NumPy)
- **Tổng hợp mã nguồn mở**: Yosys, đánh giá area/timing với OpenROAD + SKY130 PDK

## Trạng thái dự án

- [ ] Golden model Python (Conv + FC)
- [ ] RTL: PE đơn (weight-stationary)
- [ ] RTL: MAC Array (channel-parallel)
- [ ] RTL: Weight Buffer (single, sau đó ping-pong)
- [ ] RTL: Line Buffer / Window Generator + bypass cho FC
- [ ] RTL: pSum Accumulator (tap loop + channel time-multiplexing)
- [ ] RTL: Quantizer
- [ ] RTL: Top-level Controller FSM
- [ ] Verification: so khớp golden model cho cả Conv và FC
- [ ] AXI4-Lite Config Interface
- [ ] AXI4-Stream Data Interface
- [ ] RTL: Tiling Controller (nâng cao)
- [ ] Tổng hợp thử Yosys/OpenROAD (SKY130)
- [ ] Đánh giá hiệu năng: throughput, latency, PE utilization, ước lượng power/area

## Cấu trúc thư mục

```
.
├── rtl/                        # Mã nguồn SystemVerilog
├── tb/                         # Testbench
├── model/                      # Golden model Python, script train/quantize
├── sim/                        # Script mô phỏng, kịch bản test
├── synth/                      # Script tổng hợp Yosys/OpenROAD
├── docs/
│   └── explored/
│       ├── 9pe-spatial/        # Phương án đầu tiên, giữ lại làm tài liệu so sánh
│       └── systolic-gemm/      # Phương án Systolic + im2col + skew, giữ lại làm tài liệu so sánh
└── README.md
```

## Tác giả

Sinh viên ngành Kỹ thuật máy tính/Điện tử, hướng chuyên môn RTL Design & Verification.
