A=[1;1;1];
B=[1; 2; 4;]
C=B;
for i = 1:5
B=[B C.^i];
endfor
B