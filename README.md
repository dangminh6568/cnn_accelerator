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

## Đặc tả chức năng phần cứng

Đặc tả chi tiết từng khối, dùng làm checklist thiết kế RTL. Thứ tự implement khuyến nghị: PE Array → Accumulator → Line Buffer/Window → Quantizer → Controller FSM → Weight Streaming → Tiling.

### 1. Line Buffer & Window Generator

**Chức năng:**
- Lưu 3 hàng gần nhất của feature map (1 channel tại 1 thời điểm) để tạo cửa sổ trượt 3×3
- Sau mỗi chu kỳ xung nhịp, trích ra 1 cửa sổ 3×3 mới khi input dịch chuyển sang phải 1 pixel
- Xử lý padding: khi cửa sổ chạm biên ảnh/tile, chèn giá trị 0 thay vì đọc dữ liệu ngoài vùng hợp lệ

**Cần làm:**
- Buffer 3×W (W lấy từ `cfg_w`, dùng shift register hoặc dual-port BRAM tùy độ rộng ảnh tối đa hỗ trợ)
- Logic phát hiện vị trí biên (row đầu/cuối, col đầu/cuối) dựa theo `cfg_h`, `cfg_w`, `cfg_pad_en`
- Output: cửa sổ 3×3 (9 giá trị INT8) + tín hiệu valid khi cửa sổ sẵn sàng

### 2. PE Array (9 PE cố định)

**Chức năng:**
- Nhận 1 cửa sổ 3×3 (9 giá trị input) và 1 kernel 3×3 (9 giá trị weight) cùng lúc
- Thực hiện 9 phép nhân INT8×INT8 song song, sau đó cộng lại thành 1 giá trị partial sum (dùng adder tree, không cộng tuần tự để tránh delay dài)

**Cần làm:**
- 9 multiplier 8-bit×8-bit → kết quả 16-bit
- Adder tree 3 tầng (9 input → gộp dần → 1 output), độ rộng kết quả đủ lớn để không tràn (tối thiểu 20-bit)
- Thiết kế thuần tổ hợp (combinational) hoặc pipeline 1-2 tầng tùy yêu cầu tần số hoạt động

### 3. Channel Accumulator

**Chức năng:**
- Cộng dồn partial sum từ PE Array qua từng input channel (vòng lặp `ic` trong `C_in`)
- Khi xử lý xong toàn bộ input channel cho 1 output pixel, cộng thêm bias rồi chuyển kết quả sang ReLU
- Hỗ trợ trường hợp tiling: nếu input channel bị chia nhỏ qua nhiều lượt xử lý (do tile), phải đọc lại partial sum đã lưu trước đó từ bộ nhớ ngoài, cộng thêm giá trị mới, rồi ghi lại (read-modify-write) thay vì chỉ cộng dồn trong thanh ghi nội bộ

**Cần làm:**
- Thanh ghi tích lũy 32-bit, reset về 0 khi bắt đầu 1 output pixel mới (dựa theo bộ đếm `ic` chạy hết `cfg_c_in`)
- Cộng bias ở bước cuối cùng của vòng lặp input channel
- Interface đọc/ghi partial sum ra bộ nhớ ngoài cho trường hợp tiling (có thể làm ở giai đoạn sau, chưa cần ngay ở bản đầu tiên)

### 4. ReLU

**Chức năng:**
- So sánh giá trị accumulator (sau khi cộng bias) với 0, giữ nguyên nếu dương, đưa về 0 nếu âm

**Cần làm:**
- 1 bộ so sánh (MSB check nếu dùng signed) + 1 mux chọn giữa giá trị gốc và 0
- Đây là khối đơn giản nhất, có thể fuse chung tầng tổ hợp với accumulator để tiết kiệm 1 chu kỳ pipeline

### 5. Quantizer

**Chức năng:**
- Chuyển kết quả 32-bit (sau ReLU) về lại INT8 bằng cách nhân với hệ số scale rồi dịch phải (shift) một số bit nhất định
- Có clamp (giới hạn) giá trị output trong khoảng [-128, 127] hoặc [0, 255] tùy có dùng signed hay không, để tránh tràn số khi ép kiểu

**Cần làm:**
- 1 multiplier (32-bit × scale) + shifter (barrel shifter hoặc shift cố định nếu chấp nhận đơn giản hóa)
- Logic clamp sau khi shift
- Đọc `cfg_quant_scale` và `cfg_quant_shift` từ config interface — **khác nhau cho từng layer**, không hard-code

### 6. Weight Buffer & Streaming

**Chức năng:**
- Lưu trọng số (kernel 3×3) của output channel đang xử lý trong buffer on-chip
- Khi tổng weight của 1 layer vượt quá dung lượng buffer, nạp dần từng phần từ bộ nhớ ngoài qua `weight_valid/weight_ready/weight_data`
- Double buffering: trong lúc PE Array đang tính với bộ weight hiện tại, nạp trước bộ weight cho lượt tính tiếp theo (ẩn thời gian chờ nạp weight phía sau thời gian tính toán)

**Cần làm:**
- Buffer đủ chứa ít nhất 1 bộ kernel 3×3×C_in cho 1 output channel (kích thước tùy `cfg_c_in`)
- FSM quản lý 2 buffer (ping-pong), chuyển đổi khi 1 buffer đã nạp xong và buffer kia đang được dùng để tính
- *(Có thể bỏ qua double buffering ở bản đầu tiên, dùng single buffer trước cho đơn giản, thêm sau nếu còn thời gian)*

### 7. Data Reuse / Loop Order Controller

**Chức năng:**
- Quyết định thứ tự thực hiện 3 vòng lặp lồng nhau: output channel (`oc`) → vị trí pixel/tile (`oy, ox`) → input channel (`ic`)
- Mục tiêu: giảm số lần đọc/ghi dữ liệu từ bộ nhớ ngoài bằng cách tái sử dụng dữ liệu đã có trong buffer càng nhiều càng tốt trước khi nạp dữ liệu mới

**Cần làm:**
- Ở bản đầu tiên: chọn 1 thứ tự cố định đơn giản (ví dụ: giữ input tile cố định, chạy hết các output channel trước khi chuyển tile tiếp theo — tái sử dụng input, nạp lại weight)
- Ghi rõ trong tài liệu thiết kế lý do chọn thứ tự này, để phần đánh giá hiệu năng (báo cáo) có thể phân tích ưu/nhược điểm

### 8. Tiling Controller

**Chức năng:**
- Chia feature map lớn hơn buffer on-chip thành các tile nhỏ, xử lý tuần tự từng tile
- Tính vùng chồng lấp (halo) giữa các tile lân cận để pixel ở biên tile được tính đúng (cần dữ liệu từ tile kề bên)
- Quản lý địa chỉ đọc input / ghi output tương ứng với vị trí tile trong toàn bộ feature map lớn

**Cần làm:**
- Bộ đếm vị trí tile hiện tại (tile_x, tile_y)
- Logic tính offset đọc dữ liệu có overlap (đọc dư ra ngoài biên tile 1 pixel mỗi phía nếu không phải tile nằm ở rìa ảnh lớn)
- *(Khuyến nghị: làm module này tách biệt độc lập với conv engine lõi, giao tiếp qua địa chỉ/offset, để verify riêng không làm phức tạp module tính toán chính)*

### 9. Top-level Controller FSM

**Chức năng:**
- Điều phối toàn bộ pipeline: nhận config, chờ weight nạp xong, chạy vòng lặp tính toán, đồng bộ input/output stream
- Đọc đúng tham số runtime (`cfg_c_in`, `cfg_c_out`, `cfg_h`, `cfg_w`...) thay vì cố định lúc thiết kế

**Cần làm:**
- State machine tối thiểu gồm các trạng thái: `IDLE` → `LOAD_CONFIG` → `LOAD_WEIGHT` → `COMPUTE` → `DONE`
- Bộ đếm cho từng vòng lặp (`oc`, `oy`, `ox`, `ic`) đồng bộ với trạng thái `COMPUTE`
- Xuất tín hiệu `busy`, `done` theo đúng thời điểm

### 10. Config/Register Interface

**Chức năng:**
- Nhận và lưu giữ toàn bộ tham số cần thiết cho 1 lượt tính (1 layer hoặc 1 tile) trước khi bắt đầu

**Cần làm:**
- Thanh ghi lưu: `cfg_c_in`, `cfg_c_out`, `cfg_h`, `cfg_w`, `cfg_quant_scale`, `cfg_quant_shift`, `cfg_pad_en`
- Handshake `cfg_valid/cfg_ready` để nạp config trước khi `start`

---

**Gợi ý thứ tự làm việc thực tế:**

1. Viết golden model Python trước (numpy, bit-accurate INT8) — dùng làm reference cho toàn bộ các bước sau
2. RTL hóa khối 2 → 3 → 4 → 5 (PE Array → Accumulator → ReLU → Quantizer) và ghép test với 1 output pixel đơn lẻ trước, chưa cần buffer/FSM đầy đủ
3. Thêm khối 1 (Line Buffer/Window) để tự động tạo input cho khối trên theo cả feature map
4. Thêm khối 9 + 10 (Controller FSM + Config) để chạy tự động qua nhiều output channel, nhiều input channel
5. Thêm khối 6 (Weight Streaming) khi weight vượt buffer
6. Thêm khối 7, 8 (Loop order + Tiling) sau cùng — đây là phần nâng cao, có thể để làm ở giai đoạn 2 nếu thời gian hạn chế

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
