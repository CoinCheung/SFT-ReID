baseline: 
92.96, 82.57

1. loss是两个forward再统一做一个backward
92.43, 82.21 - 差了一点点

2. 使用288x144再crop的aug
93.67, 83.19 - 提高了半个点
93.46, 83.34
93.14, 83.24
93.52, 83.21

3. 加post
92.96, 87.17
92.75, 87.40

4. share bottleneck loss
2 forward 1 backward
93.11, 83.46/92.8, 87.45

3. warmup 使用iter的

4. 使用cosine lr

5. 使用smooth label 
