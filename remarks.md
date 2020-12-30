# 备注

## 一些系统参数

以下是一些参数，应该在整个系统运行前确定，后续可以考虑封装到一个方法中
- Classification/Regression：分类模型还是回归模型，
    决定之后使用Class-based distance还是MAD-based distance
- 阈值T：当某一个输入对应输出的distance超过T时，认为此输入会造成两个模型的不一致性  
    Class-based distance：1 <= T <= pow(2, k-1)  
    MAD-based distance：0 <= T <= 1
- 阈值p：对于两个不同backend的模型，当验证集中至少p%的输入造成不一致时，认为backends具有不一致性