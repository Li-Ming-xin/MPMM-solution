1 matmul.cu

2 matvec.cu

3.a 512 threads per block.  
3.b (19 * 5) * (16 * 32) = 48640 threads in the grid.  
3.c 19 * 5 = 95 blocks in the grid.  
3.d 150 * 300 = 45000 threads execute the code on line 05.

4.a (20 - 1) * 400 + (10 - 1) = 7609  
4.b (10 - 1) * 500 + (20 - 1) = 4519

5 (5 - 1) * 500 * 400 + (20 - 1) * 400 + (10 - 1) = 807609
