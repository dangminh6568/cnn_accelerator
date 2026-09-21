# Microarchitecture Specification — Conv2D Accelerator Core (CONVX)

**Version:** 0.1 (Draft cho RTL implementation)
**Loại tài liệu:** Microarchitecture Spec
**Trạng thái:** Sẵn sàng để code RTL — các mục còn mở được đánh dấu ở mục 10.

---

## Cách dùng tài liệu này

Tài liệu này mô tả kiến trúc ở mức đủ chi tiết để bạn bắt đầu viết RTL trực tiếp: có port list, register map, công thức địa chỉ bộ nhớ, bảng trạng thái FSM và một ví dụ cycle-by-cycle. Thứ tự code RTL khuyến nghị (bottom-up):

1. `pe.v` → `pe_array.v` → `adder_tree.v`
2. `accumulator_bank.v`
3. `post_proc.v`
4. `line_buffer.v`, `weight_mem.v`, `bias_mem.v`
5. `cfg_reg_block.v`, `output_fifo.v`
6. `control_fsm.v` (ghép tất cả lại)
7. `conv_accel_top.v`

---

## 1. Tổng quan & Phạm vi (Scope)

CONVX là một accelerator thực hiện phép **Conv2D lượng tử hóa INT8** cho một layer tại một thời điểm, bao gồm chuỗi xử lý: `MAC → Bias Add → ReLU → Requantize`.

### 1.1 Bảng phạm vi hỗ trợ (Scope Lock)

| Thông số | Giá trị hỗ trợ |
|---|---|
| Kernel size (K) | 1×1, 3×3, 5×5 |
| Stride | 1, 2 |
| Padding mode | `VALID` (không pad), `SAME` (zero-pad giữ nguyên kích thước output theo stride) |
| Input channels (Cin) | 1 – 32 |
| Output channels (Cout) | 1 – 32 |
| Feature map H/W | 1 – 256 |
| Data type input/weight | INT8 (signed, 2's complement) |
| Data type accumulator | INT32 (signed) |
| Data type bias | INT32 (signed) |
| Data type output | INT8 (signed, sau requantize + saturate) |

Mọi giá trị ngoài phạm vi trên coi là **không được định nghĩa hành vi** (out of scope cho v0.1).

---

## 2. Kiến trúc tổng thể

```
                        ┌────────────────────────────┐
   cfg_* ──────────────▶│      Config Reg Block       │
                        └──────────────┬─────────────┘
                                       │ (kernel_size, stride, pad_mode,
                                       │  Cin, Cout, H, W, requant_shift)
                                       ▼
 wload_* ──────────▶┌───────────┐    ┌───────────────────────────┐
                    │ Weight Mem │◀───┤                             │
 bload_* ──────────▶│ + Bias Mem │    │                             │
                    └─────┬──────┘    │      Control FSM           │
                          │           │   (đếm oy, ox, cout_tile,   │
 in_valid/in_ready/  ┌────▼──────┐    │    ky, kx, cin_tile)        │
 in_data ───────────▶│Line Buffer│◀───┤                             │
                    └─────┬──────┘    └──────────┬──────────────────┘
                          │                      │ điều khiển
                          ▼                      ▼
                    ┌─────────────────────────────────┐
                    │      PE Array (CIN_TILE ×         │
                    │        COUT_TILE MACs)            │
                    └─────────────┬───────────────────┘
                                  ▼
                    ┌─────────────────────┐
                    │   Adder Tree (per    │
                    │   output channel)     │
                    └─────────────┬───────┘
                                  ▼
                    ┌─────────────────────┐
                    │  Accumulator Bank    │
                    │   (COUT_TILE reg)    │
                    └─────────────┬───────┘
                                  ▼
                    ┌─────────────────────┐
                    │ Post-Proc: +Bias →   │
                    │ ReLU → Requant/Sat   │
                    └─────────────┬───────┘
                                  ▼
                    ┌─────────────────────┐
 out_valid/out_ready│   Output FIFO/Writer │
 out_data ◀─────────┤                       │
                    └─────────────────────┘
```

### 2.1 Nguyên lý dataflow

- **Kiến trúc này là Output-Stationary (OS) thuần túy**: accumulator (ứng với 1 vị trí output tại pixel×cout_tile) được giữ cố định và tích lũy kết quả trong suốt toàn bộ vòng lặp kernel taps + cin_tiles. Weight và activation được đọc mới hoàn toàn ở mỗi tap và không được giữ lại để tái sử dụng qua các cycle kế tiếp — vì vậy thiết kế **không** có yếu tố weight-stationary (khác với kiến trúc WS, nơi 1 weight được nạp 1 lần rồi dùng cho nhiều activation liên tiếp trước khi đổi weight).
- **Streaming theo từng pixel** (không tile theo không gian): nhờ line buffer chỉ giữ K dòng gần nhất, không cần buffer toàn bộ feature map → tiết kiệm bộ nhớ, và không phát sinh vấn đề "kích thước không chia hết cho PE array" ở chiều không gian.

### 2.2 Vòng lặp tính toán (pseudocode tham chiếu)

```
for oy in [0, out_height):
  for ox in [0, out_width):
    for cout_tile in [0, num_cout_tiles):
      reset acc[0..COUT_TILE-1] = 0
      for ky in [0, kernel_size):
        for kx in [0, kernel_size):
          for cin_tile in [0, num_cin_tiles):
            # đọc CIN_TILE input activations tại (oy*stride+ky-pad, ox*stride+kx-pad)
            # đọc khối weight CIN_TILE x COUT_TILE tại (cout_tile, ky, kx, cin_tile)
            # PE array: product[i][j] = act[i] * weight[i][j]
            # adder tree: partial_sum[j] = Σ_i product[i][j]
            acc[j] += partial_sum[j]   # với mọi j trong COUT_TILE
      # post-process
      for j in [0, COUT_TILE):
        out[j] = saturate_int8( relu( acc[j] + bias[cout_tile*COUT_TILE + j] ) >> requant_shift )
      push out[] ra output interface (chỉ các lane hợp lệ nếu là group cuối)
```

> Vì `cout_tile` nằm **trong** vòng lặp `(oy, ox)`, mỗi pixel được xử lý xong hoàn toàn (mọi output channel) trước khi sang pixel kế — chỉ cần line buffer K dòng, không cần buffer cả ảnh.

---

## 3. Tham số hóa (Parameterization)

### 3.1 Compile-time parameters (Verilog `parameter`)

| Tên | Giá trị mặc định | Ý nghĩa |
|---|---|---|
| `CIN_TILE` | 8 | Số input channel xử lý song song (số hàng PE array) |
| `COUT_TILE` | 8 | Số output channel xử lý song song (số cột PE array) |
| `K_MAX` | 5 | Kernel size lớn nhất hỗ trợ |
| `ACT_WIDTH` | 8 | Bit width activation/weight |
| `ACC_WIDTH` | 32 | Bit width accumulator |
| `MAX_CIN` | 32 | Input channel tối đa |
| `MAX_COUT` | 32 | Output channel tối đa |
| `MAX_DIM` | 256 | H/W tối đa (→ counter 10 bit) |

### 3.2 Runtime-configurable (qua register interface, xem mục 6.1)

`kernel_size`, `stride`, `pad_mode`, `in_channels`, `out_channels`, `in_height`, `in_width`, `requant_shift`.

### 3.3 Giá trị dẫn xuất (tính 1 lần khi config, lưu vào thanh ghi nội bộ)

```
num_cin_tiles  = ceil(in_channels  / CIN_TILE)
num_cout_tiles = ceil(out_channels / COUT_TILE)

# VALID:
out_h = floor((in_height - kernel_size) / stride) + 1
out_w = floor((in_width  - kernel_size) / stride) + 1

# SAME:
out_h = ceil(in_height / stride)
out_w = ceil(in_width  / stride)
pad_total_h = max((out_h - 1) * stride + kernel_size - in_height, 0)
pad_total_w = max((out_w - 1) * stride + kernel_size - in_width, 0)
pad_top  = pad_total_h / 2   (floor)
pad_left = pad_total_w / 2   (floor)
```

---

## 4. Định dạng dữ liệu (Data Types & Numeric Formats)

| Tín hiệu | Width | Format | Range |
|---|---|---|---|
| Input activation | 8 bit | INT8 signed | [-128, 127] |
| Weight | 8 bit | INT8 signed | [-128, 127] |
| PE product | 16 bit | INT16 signed | [-16384, 16384] |
| Adder tree sum (1 tap, CIN_TILE lanes) | 20 bit | INT20 signed | đủ cho CIN_TILE=8 × max product |
| Accumulator | 32 bit | INT32 signed | đủ cho K_MAX²×MAX_CIN = 25×32 = 800 MAC tối đa |
| Bias | 32 bit | INT32 signed | theo layer |
| Output (sau requant) | 8 bit | INT8 signed | [-128, 127] |

**Công thức requantize:**
```
sum      = acc + bias                       # INT32
relu_out = (sum < 0) ? 0 : sum               # INT32, không âm
shifted  = relu_out >> requant_shift         # logical shift (relu_out không âm)
out_i8   = (shifted > 127) ? 127 : shifted[7:0]
```

---

## 5. Đặc tả module

### 5.1 Config Register Block

Giao diện ghi/đọc đơn giản kiểu APB-lite: `cfg_addr[7:0]`, `cfg_wdata[31:0]`, `cfg_wen`, `cfg_rdata[31:0]`, `cfg_ren`.

| Địa chỉ | Tên | Bit field | Access | Mô tả |
|---|---|---|---|---|
| 0x00 | CTRL | [0]=START, [1]=SOFT_RESET | W | Ghi 1 vào START để bắt đầu (tự clear) |
| 0x04 | STATUS | [0]=BUSY, [1]=DONE, [2]=ERROR | R | DONE giữ nguyên đến lần START kế |
| 0x08 | LAYER_CFG0 | [2:0]=kernel_size, [4:3]=stride, [5]=pad_mode | RW | pad_mode: 0=VALID, 1=SAME |
| 0x0C | LAYER_CFG1 | [7:0]=in_channels, [15:8]=out_channels | RW | |
| 0x10 | LAYER_CFG2 | [9:0]=in_height, [19:10]=in_width | RW | |
| 0x14 | REQUANT_CFG | [4:0]=requant_shift | RW | |

**Hành vi ERROR:** set khi `kernel_size ∉ {1,3,5}`, `stride ∉ {1,2}`, hoặc `in_channels/out_channels` vượt `MAX_CIN/MAX_COUT`. Khi ERROR=1, FSM không chuyển sang LOAD_WEIGHT.

### 5.2 Weight Memory & Loader

- **Tổ chức bộ nhớ:** 1 word = toàn bộ khối trọng số CIN_TILE×COUT_TILE cho **một** tap `(cout_tile, ky, kx, cin_tile)`.
- **Word width** = `CIN_TILE × COUT_TILE × ACT_WIDTH` = 8×8×8 = **512 bit**.
- **Depth** = `(MAX_COUT/COUT_TILE) × K_MAX² × (MAX_CIN/CIN_TILE)` = 4×25×4 = **400 word**.
- **Công thức địa chỉ** (dùng chung cho cả load lẫn compute — đảm bảo thứ tự stream nạp trùng thứ tự đọc):
  ```
  weight_addr = ((cout_tile_idx * kernel_size + ky) * kernel_size + kx) * num_cin_tiles + cin_tile_idx
  ```
- **Giao diện nạp:** `wload_valid`, `wload_ready`, `wload_data[511:0]`, `wload_last`. Phần mềm/testbench phải gửi đúng thứ tự: vòng ngoài `cout_tile` → `ky` → `kx` → `cin_tile` (vòng trong cùng), mỗi word 512 bit chứa `weight[cin_local][cout_local]` được đóng gói: `wload_data[(cin_local*COUT_TILE+cout_local)*8 +: 8]`.
- **Zero-padding kênh:** nếu `in_channels` hoặc `out_channels` không chia hết `CIN_TILE`/`COUT_TILE`, phần mềm phải zero-pad tensor trọng số ở các lane dư trước khi nạp — phần cứng không cần xử lý gì thêm cho MAC (0×x=0).

### 5.3 Bias Memory

- Depth = `MAX_COUT` = 32, width = 32 bit.
- Giao diện: `bload_valid`, `bload_ready`, `bload_data[31:0]`, `bload_last`. Nạp tuần tự theo `cout` index 0→out_channels-1.

### 5.4 Line Buffer (Input)

- Gồm **K_MAX** row-buffer riêng biệt (dùng như circular buffer theo `ky`), mỗi row-buffer: depth = `MAX_DIM` (256), width = `MAX_CIN × ACT_WIDTH` = 256 bit (toàn bộ channel của 1 pixel/word).
- **Giao diện nạp:** `in_valid`, `in_ready`, `in_data[255:0]` — mỗi cycle nhận 1 pixel (tất cả channel), producer bên ngoài chịu trách nhiệm gửi đúng thứ tự raster (row-major), và zero-pad các channel lane vượt `in_channels`.
- **Đọc trong compute:** tại tap `(ky, kx, cin_tile)`, đọc row buffer tương ứng `ky`, cột `ox*stride+kx-pad_left`, rồi lấy slice `CIN_TILE` channel:
  ```
  cin_slice = row_buf[ky][col_idx][ (cin_tile_idx*CIN_TILE)*8 +: CIN_TILE*8 ]
  ```
- **Xử lý padding:** trước khi đọc, kiểm tra:
  ```
  row_idx = oy*stride + ky - pad_top
  col_idx = ox*stride + kx - pad_left
  is_pad  = (row_idx < 0) || (row_idx >= in_height) || (col_idx < 0) || (col_idx >= in_width)
  ```
  Nếu `is_pad = 1` → toàn bộ CIN_TILE lane activation = 0 (bypass đọc buffer, không cần buffer chứa dữ liệu ảo).

### 5.5 PE Array

- Kích thước: `CIN_TILE × COUT_TILE` = 8×8 = 64 PE.
- Mỗi PE: 1 thanh ghi weight (nạp từ weight mem mỗi tap), 1 bộ nhân 8×8→16 bit signed.
  ```
  product[i][j] = act[i] * weight[i][j]     # i = 0..CIN_TILE-1, j = 0..COUT_TILE-1
  ```
- **Wiring:** `act[i]` broadcast theo hàng `i` cho toàn bộ COUT_TILE cột; `weight[i][j]` được nạp đúng vào vị trí PE(i,j) cho tile hiện tại và chỉ dùng trong đúng 1 cycle MAC của tap đó — **không** giữ lại cho tap kế tiếp (không phải weight-stationary). Khác với systolic array vốn dịch chuyển dữ liệu qua từng PE theo từng cycle, ở đây dữ liệu chỉ broadcast + tính tại chỗ trong 1 cycle, giúp đơn giản hóa control.

### 5.6 Adder Tree

- Với mỗi cột `j`, cộng dồn `CIN_TILE` product thành 1 giá trị: cây cộng nhị phân độ sâu `log2(CIN_TILE)` = 3 tầng (cho CIN_TILE=8).
- Output width = 20 bit signed (đủ margin).
- Có thể pipeline 1–2 tầng nếu timing không đạt — ghi rõ số stage pipeline thực tế trong RTL comment vì nó ảnh hưởng đến số cycle mỗi tap trong FSM.

### 5.7 Accumulator Bank

- `COUT_TILE` thanh ghi 32-bit signed.
- Reset về 0 khi bắt đầu 1 pixel×cout_tile mới (state `COMPUTE_INIT`).
- Cộng dồn: `acc[j] <= acc[j] + adder_tree_out[j]` mỗi khi hoàn thành 1 tap (1 lần MAC cho toàn bộ CIN_TILE, COUT_TILE).

### 5.8 Post-Processing Unit

Pipeline 3 tầng, áp dụng song song cho COUT_TILE lane:

| Stage | Phép toán |
|---|---|
| 1. ADD_BIAS | `sum[j] = acc[j] + bias[cout_tile*COUT_TILE+j]` |
| 2. RELU | `relu_out[j] = (sum[j] < 0) ? 0 : sum[j]` |
| 3. REQUANT_SAT | `out[j] = (relu_out[j]>>requant_shift > 127) ? 127 : (relu_out[j]>>requant_shift)[7:0]` |

### 5.9 Output FIFO/Writer

- Đóng gói `COUT_TILE` giá trị INT8 thành `out_data[63:0]`.
- Giao thức `out_valid`/`out_ready` chuẩn: giữ `out_valid=1` và giữ nguyên `out_data` cho đến khi `out_ready=1` được lấy (không được đổi dữ liệu khi đang chờ).
- **Group cuối không đủ COUT_TILE:** thêm output `out_valid_mask[COUT_TILE-1:0]` (hoặc field riêng) đánh dấu lane nào hợp lệ, dựa trên `valid_count = out_channels - cout_tile_idx*COUT_TILE` khi `valid_count < COUT_TILE`.

### 5.10 Control FSM

| State | Điều kiện vào | Hành động trong state | Điều kiện ra |
|---|---|---|---|
| `S_IDLE` | reset / sau DONE | chờ `CTRL.START=1` | START=1 & ERROR=0 → `S_CALC_DERIVED` |
| `S_CALC_DERIVED` | — | tính `num_cin_tiles, num_cout_tiles, out_h, out_w, pad_top, pad_left` | luôn → `S_LOAD_WEIGHT` |
| `S_LOAD_WEIGHT` | — | nhận `wload_*`, ghi vào weight mem theo địa chỉ tăng dần | `wload_last=1` → `S_LOAD_BIAS` |
| `S_LOAD_BIAS` | — | nhận `bload_*`, ghi vào bias mem | `bload_last=1` → `S_COMPUTE_INIT` |
| `S_COMPUTE_INIT` | pixel/cout_tile mới | reset `acc[]=0`, reset `ky_cnt=kx_cnt=cin_tile_cnt=0` | luôn → `S_FETCH` |
| `S_FETCH` | — | tính địa chỉ weight/line-buffer, đọc dữ liệu (1 cycle latency SRAM) | luôn → `S_MAC` |
| `S_MAC` | — | PE array nhân + adder tree + cộng dồn accumulator | luôn → `S_LOOP_CTRL` |
| `S_LOOP_CTRL` | — | tăng `cin_tile_cnt`; nếu hết → tăng `kx_cnt`; nếu hết → tăng `ky_cnt` | còn tap → `S_FETCH`; hết tap → `S_POST` |
| `S_POST` | — | pipeline bias+relu+requant (3 cycle) | luôn (sau 3 cycle) → `S_OUTPUT` |
| `S_OUTPUT` | — | đẩy `out_data`, chờ `out_ready` | `out_ready=1` → `S_NEXT_PIXEL` |
| `S_NEXT_PIXEL` | — | tăng `cout_tile_cnt`; nếu hết → reset về 0, tăng `ox_cnt`; nếu hết → reset, tăng `oy_cnt` | còn → `S_COMPUTE_INIT`; hết `oy_cnt` → `S_DONE` |
| `S_DONE` | — | set `STATUS.DONE=1`, `BUSY=0` | luôn → `S_IDLE` |

**Bộ đếm cần thiết** (đặt trong `control_fsm.v`):

| Tên | Width | Miền giá trị |
|---|---|---|
| `oy_cnt` | 10 bit | 0 .. out_h-1 |
| `ox_cnt` | 10 bit | 0 .. out_w-1 |
| `cout_tile_cnt` | 2 bit | 0 .. num_cout_tiles-1 |
| `ky_cnt` | 3 bit | 0 .. kernel_size-1 |
| `kx_cnt` | 3 bit | 0 .. kernel_size-1 |
| `cin_tile_cnt` | 2 bit | 0 .. num_cin_tiles-1 |

---

## 6. Port List (top-level `conv_accel_top`)

| Tên | Hướng | Width | Mô tả |
|---|---|---|---|
| `clk` | in | 1 | Clock |
| `rst_n` | in | 1 | Active-low reset |
| `cfg_addr` | in | 8 | Địa chỉ thanh ghi config |
| `cfg_wdata` | in | 32 | Dữ liệu ghi |
| `cfg_wen` | in | 1 | Write enable |
| `cfg_rdata` | out | 32 | Dữ liệu đọc |
| `cfg_ren` | in | 1 | Read enable |
| `wload_valid` | in | 1 | Weight stream valid |
| `wload_ready` | out | 1 | Weight stream ready |
| `wload_data` | in | 512 | 1 tile trọng số CIN_TILE×COUT_TILE |
| `wload_last` | in | 1 | Đánh dấu word cuối |
| `bload_valid` | in | 1 | Bias stream valid |
| `bload_ready` | out | 1 | Bias stream ready |
| `bload_data` | in | 32 | 1 giá trị bias |
| `bload_last` | in | 1 | Đánh dấu word cuối |
| `in_valid` | in | 1 | Input pixel valid |
| `in_ready` | out | 1 | Input pixel ready |
| `in_data` | in | 256 | Toàn bộ channel của 1 pixel |
| `out_valid` | out | 1 | Output valid |
| `out_ready` | in | 1 | Output ready (backpressure) |
| `out_data` | out | 64 | COUT_TILE giá trị INT8 |
| `out_valid_mask` | out | 8 | Lane nào hợp lệ (group cuối) |
| `busy` | out | 1 | Đang tính toán |
| `done` | out | 1 | Hoàn thành 1 layer |
| `error` | out | 1 | Config lỗi |

**Quy tắc handshake valid/ready:** chuẩn AXI-Stream — bên gửi giữ `valid` và dữ liệu ổn định cho đến khi thấy `ready=1` tại cạnh clock; không được rút `valid` khi chưa được `ready` chấp nhận.

---

## 7. Xử lý edge case (tóm tắt tham chiếu)

| Edge case | Cách xử lý |
|---|---|
| Padding biên ảnh | Line buffer trả activation = 0 khi `is_pad=1` (mục 5.4), không cần lưu dữ liệu ảo |
| Cin/Cout không chia hết tile | Phần mềm zero-pad weight tensor; output dùng `out_valid_mask` để đánh dấu lane hợp lệ ở group cuối |
| Overflow accumulator | ACC_WIDTH=32 đủ margin cho phạm vi scope (mục 4); post-proc bắt buộc saturate về INT8 sau requant |
| Backpressure output | FSM dừng ở `S_OUTPUT` khi `out_ready=0`, giữ nguyên `acc`/`out_data`, không mất dữ liệu |
| Input chưa sẵn sàng | `in_ready` chỉ bật khi FSM đang ở trạng thái cần đọc line buffer mới (tách biệt với giai đoạn compute nội bộ) — cần thiết kế thêm 1 prefetch state nếu muốn pipeline sâu hơn (xem mục 10) |

---

## 8. Worked Example — Cycle-level Walkthrough

**Cấu hình ví dụ (để bảng ngắn gọn):** `kernel_size=3, stride=1, pad_mode=VALID, in_channels=8 (=CIN_TILE → num_cin_tiles=1), out_channels=8 (=COUT_TILE → num_cout_tiles=1), in_height=5, in_width=5` → `out_h=out_w=3`.

Theo dõi **pixel đầu tiên** `(oy=0, ox=0)`:

| Cycle | State | ky,kx,cin_tile | Hoạt động |
|---|---|---|---|
| 1 | S_COMPUTE_INIT | – | acc[0..7]=0, ky=kx=cin_tile=0 |
| 2 | S_FETCH | 0,0,0 | đọc weight_addr=0, đọc line_buf[ky=0][col=0] |
| 3 | S_MAC | 0,0,0 | PE array 8×8 MAC, adder tree → acc += |
| 4 | S_LOOP_CTRL | →0,1,0 | cin_tile hết (num_cin_tiles=1) → tăng kx |
| 5 | S_FETCH | 0,1,0 | weight_addr=1, line_buf[0][col=1] |
| 6 | S_MAC | 0,1,0 | acc += |
| ... | ... | ... | lặp lại cho (0,2,0), (1,0,0)...(1,2,0), (2,0,0)...(2,2,0) — tổng **9 tap** |
| ~26 | S_LOOP_CTRL | hết ky=2,kx=2 | → S_POST |
| 27-29 | S_POST | – | +bias → relu → requant (3 cycle pipeline) |
| 30 | S_OUTPUT | – | out_valid=1, out_data=8×INT8, chờ out_ready |
| 31 | S_NEXT_PIXEL | – | cout_tile hết (num_cout_tiles=1) → ox=1 → S_COMPUTE_INIT |

Tổng cộng ~30 cycle/pixel trong ví dụ này (chưa pipeline tối ưu) × 9 pixel ≈ 270 cycle cho cả layer — con số này chính là baseline để bạn so sánh khi tối ưu pipeline sau này (mục tiêu synthesis/PPA đã bàn ở bước trước).

---

## 9. Danh sách file RTL đề xuất

```
rtl/
├── conv_accel_top.v        # ghép nối toàn bộ
├── cfg_reg_block.v
├── weight_mem.v
├── bias_mem.v
├── line_buffer.v
├── pe.v                     # 1 PE
├── pe_array.v               # CIN_TILE x COUT_TILE instance của pe.v
├── adder_tree.v
├── accumulator_bank.v
├── post_proc.v
├── output_fifo.v
└── control_fsm.v
```

---

## 10. Giả định & mục còn mở (Assumptions & Open Items)

- Giả định weight mem / bias mem / line buffer được infer thành SRAM/BRAM 1-cycle read latency (đồng bộ). Nếu công cụ synthesis không tự infer đúng, cần viết lại theo template BRAM của target cụ thể.
- `S_MAC` hiện giả định adder tree tổ hợp (combinational) trong 1 cycle — nếu không đạt timing khi synthesis, cần pipeline thêm và cập nhật lại bảng FSM/cycle count.
- Chưa tối ưu double-buffering weight (nạp weight layer kế trong lúc layer hiện tại đang compute) — đây là điểm có thể nêu như "future work" khi trình bày dự án.
- `in_ready` hiện đơn giản hóa (gắn với state cần dữ liệu mới); nếu muốn prefetch/pipeline sâu hơn giữa các pixel, cần thiết kế thêm logic double-buffer cho line buffer — nêu rõ trong RTL comment nếu bạn chọn đơn giản hóa ở v0.1.
- Tài liệu **Verification Plan** (golden model, testbench, coverage) sẽ là bước tiếp theo, dùng chính port list và FSM ở đây làm cơ sở xây driver/monitor/scoreboard.
