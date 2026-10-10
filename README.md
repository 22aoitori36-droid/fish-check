# 魚判定

水族館で撮った魚の写真から、種類を推定するスマホ向けウェブアプリです。
判定はすべて端末の中で行います（写真はどこにも送信されません）。

- 初回のみ AI のデータ（約330MB）をダウンロードします。2回目以降はオフラインでも動きます。
- 「どこの水族館？」で海遊館・京都水族館を選ぶと、その館の展示種にしぼって判定します。

## 使っているもの
- 画像認識モデル: [BioCLIP 2](https://huggingface.co/imageomics/bioclip-2)（Imageomics Institute）を int8 に量子化して使用
- 推論: [ONNX Runtime Web](https://onnxruntime.ai/)
- 参考写真: [iNaturalist](https://www.inaturalist.org/) の CC ライセンス写真（撮影者のクレジットを画面に表示）
- 和名: Wikidata、豆知識: Wikipedia・各水族館サイトなどを参照して作成
