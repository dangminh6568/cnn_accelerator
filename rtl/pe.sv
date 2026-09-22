module pe #(
    parameter DATA_WIDTH   = 8,
    parameter RESULT_WIDTH = 16
)(
    input  logic                            clk,
    input  logic                            rst_n,

    input  logic                            valid_in,
    input  logic signed [DATA_WIDTH-1:0]    weight_in,
    input  logic signed [DATA_WIDTH-1:0]    data_in,

    output logic signed [RESULT_WIDTH-1:0]  data_out,
    output logic                            valid_out
);

    logic signed [RESULT_WIDTH-1:0] result;

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            result    <= '0;
            valid_out <= 1'b0;
        end else begin
            if (valid_in) begin
                result <= weight_in * data_in;
            end 
            
            valid_out <= valid_in;
        end
    end

    assign data_out = result;

endmodule