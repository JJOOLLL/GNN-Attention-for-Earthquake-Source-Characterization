# Method sources / 方法来源

The common framework builds on van den Ende & Ampuero (2020). The no-message/max-pooling configuration represents that baseline within this experimental pipeline, rather than a byte-for-byte redistribution of their original repository. Academic attribution does not change third-party software licenses.

共同框架基于 van den Ende & Ampuero (2020)。无消息传递＋最大池化配置在本文实验流程中代表该基线，并非原仓库逐文件原样发行。学术引用不改变第三方软件许可证。

- van den Ende, M. P. & Ampuero, J. P. (2020). Automated seismic source characterization using deep graph neural networks. *Geophysical Research Letters*, 47, e2020GL088690.
- Kipf, T. N. & Welling, M. (2017). Semi-supervised classification with graph convolutional networks. *International Conference on Learning Representations*.
- Münchmeyer, J., Bindi, D., Leser, U. & Tilmann, F. (2021). Earthquake magnitude and location estimation from real-time seismic waveforms with a transformer network. *Geophysical Journal International*, 226, 1086–1104. (TEAM-LM)
- Zhang, X., Reichard-Flynn, W., Zhang, M., Hirn, M. & Lin, Y. (2022). Spatiotemporal graph convolutional networks for earthquake source characterization. *Journal of Geophysical Research: Solid Earth*, 127, e2022JB024401. (STGNN)
- Beyreuther, M., Barsch, R., Krischer, L., Megies, T., Behr, Y. & Wassermann, J. (2010). ObsPy: A Python toolbox for seismology. *Seismological Research Letters*, 81, 530–533.

TEAM-LM/STGNN implementations are not bundled. PyTorch, PyTorch Geometric, ObsPy, and other dependencies are installed separately under their own terms.

本包不打包 TEAM-LM/STGNN 模型实现。PyTorch、PyTorch Geometric、ObsPy 等依赖单独安装，遵循各自许可。
