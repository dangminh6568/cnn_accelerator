module pe_array #(
    parameter CIN_TILE     = 8,      // row
    parameter COUT_TILE    = 8,      // column
    parameter DATA_WIDTH   = 8,
    parameter RESULT_WIDTH = 16
)(
    input  logic                            clk,
    input  logic                            rst_n,

    input  logic                            valid_in,
    input  logic signed [DATA_WIDTH-1:0]    act_in    [CIN_TILE-1:0],
    input  logic signed [DATA_WIDTH-1:0]    weight_in [CIN_TILE-1:0][COUT_TILE-1:0],

    output logic signed [RESULT_WIDTH-1:0]  product_out [CIN_TILE-1:0][COUT_TILE-1:0],
    output logic                            valid_out
);

    
    logic valid_out_mat [CIN_TILE-1:0][COUT_TILE-1:0];

    genvar i, j;
    generate
        for (i = 0; i < CIN_TILE; i++) begin : row
            for (j = 0; j < COUT_TILE; j++) begin : col
                pe #(
                    .DATA_WIDTH   (DATA_WIDTH),
                    .RESULT_WIDTH (RESULT_WIDTH)
                ) u_pe (
                    .clk       (clk),
                    .rst_n     (rst_n),
                    .valid_in  (valid_in),
                    .weight_in (weight_in[i][j]),
                    .data_in   (act_in[i]),        
                    .data_out  (product_out[i][j]),
                    .valid_out (valid_out_mat[i][j])
                );
            end
        end
    endgenerate

  
    assign valid_out = valid_out_mat[0][0];

endmodule