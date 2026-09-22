module adder_tree #(
    parameter CIN_TILE       = 8,      // row
    parameter COUT_TILE      = 8,      // column
    parameter RESULT_WIDTH   = 16,
    parameter ADDER_WIDTH    = 20
) (
    input logic  clk,
    input logic  rst_n,

    input logic  valid_in,
    input logic  signed [RESULT_WIDTH-1:0]  pe_in [CIN_TILE-1:0][COUT_TILE-1:0],

    output logic signed [ADDER_WIDTH-1:0]  tree_out [COUT_TILE-1:0],
    output logic valid_out
);


logic signed [RESULT_WIDTH:0]   stg1_sum [3:0][COUT_TILE-1:0];
logic valid_stg1;

logic signed [RESULT_WIDTH+1:0] stg2_sum [1:0][COUT_TILE-1:0];
logic valid_stg2;


always_ff @(posedge clk or negedge rst_n) begin
    if(!rst_n)begin
        valid_stg1 <= 0;
        valid_stg2 <= 0;
        valid_out  <= 0;
        for (int j = 0; j < COUT_TILE; j++) begin
            for (int i = 0; i < 4; i++) stg1_sum[i][j] <= '0;
            for (int i = 0; i < 2; i++) stg2_sum[i][j] <= '0;
            tree_out[j] <= '0;
        end
    end else begin
        if(valid_in)begin
            for(int j=0; j < COUT_TILE; j++)begin
                stg1_sum [0][j] <= pe_in [0][j] + pe_in [1][j];
                stg1_sum [1][j] <= pe_in [2][j] + pe_in [3][j];
                stg1_sum [2][j] <= pe_in [4][j] + pe_in [5][j];
                stg1_sum [3][j] <= pe_in [6][j] + pe_in [7][j];
            end   
        end
        valid_stg1 <= valid_in;  

        if(valid_stg1)begin
            for(int j=0; j < COUT_TILE; j++)begin
                stg2_sum [0][j] <= stg1_sum [0][j] + stg1_sum [1][j];
                stg2_sum [1][j] <= stg1_sum [2][j] + stg1_sum [3][j];
            end
        end
        valid_stg2 <= valid_stg1;

        if(valid_stg2)begin
            for(int j=0; j < COUT_TILE; j++)begin
                tree_out[j] <= stg2_sum [0][j] + stg2_sum [1][j];
            end
        end
        valid_out <= valid_stg2;
    end 
end






endmodule