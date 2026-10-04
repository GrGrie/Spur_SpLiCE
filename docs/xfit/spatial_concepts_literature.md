# Spatial concepts: literature pass

Reviewed 2026-10-04 against primary sources (arXiv, OpenReview, CVF, ICLR and NeurIPS proceedings, official repositories). Every number below was read in the paper itself. Where a number came from an automated reading of an HTML page and could not be cross-checked in the PDF, the entry says so. "WGA" means worst-group accuracy.

Context: ResNet-18 trained from scratch with SimCLR on MetaShift, Waterbirds, Spur-CIFAR10 and CelebA, evaluated with a logistic probe on a small group-balanced subset. Image-level concept distillation from CLIP failed because the student learned only the component that correlated concepts share. The spatial direction localizes concepts with frozen CLIP ViT-B/32 (7x7 patch grid at 224 px) and either distils per-location concept maps into the 512x7x7 ResNet feature map or builds concept-guided counterfactual views.

---

## 1. Training-free dense CLIP localization

### Summary table (mIoU, %, numbers as printed by each paper)

| Method | Key change to the last block(s) | Backbone in main table | Input protocol | VOC21 | PC60 | COCO-Obj | VOC20 | PC59 | COCO-Stuff | City | ADE | ViT-B/32 reported? |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MaskCLIP (as re-run by SCLIP) | value path only, no q/k | ViT-B/16 | 336 short side, 224 window, stride 112 | 43.4 | 23.2 | 20.6 | 74.9 | 26.4 | 16.7 | 24.9 | 11.9 | no |
| SCLIP | correlative self-attention | ViT-B/16 | same | 59.1 | 30.4 | 30.5 | 80.4 | 34.2 | 22.4 | 32.2 | 16.1 | no |
| ClearCLIP | q-q attention, no residual, no FFN | ViT-B/16 | 448 short side, sliding window | 51.8 | 32.6 | 33.0 | 80.9 | 35.9 | 23.9 | 30.0 | 16.7 | no |
| NACLIP | k-k attention + Gaussian neighbourhood prior, no residual, no FFN | ViT-B/16 | 336 short side, 224 window, stride 112 | 58.9 | 32.2 | 33.2 | - | - | - | 35.5 | 17.4 | yes |
| GEM | parallel self-self (q-q, k-k, v-v ensemble) path | ViT-B/16 | 448 short side | 46.2 (VOC) | 32.6 (Context) | - | - | - | - | - | 15.7 | yes |

Each paper uses its own protocol and its own re-run of the baselines, so numbers are comparable within a row group only. ClearCLIP reports SCLIP at 51.4 on VOC21 under its 448 protocol, while SCLIP reports itself at 59.1 under 336.

### MaskCLIP (Zhou, Loy and Dai, ECCV 2022)
- Claim: CLIP's dense features already align with the text space once the last attention pooling is bypassed. [arXiv 2112.01071](https://arxiv.org/abs/2112.01071)
- Method: drop the query and key embeddings of the last attention layer and turn the value embedding and the final linear layer into two 1x1 convolutions; the text embeddings act as a per-pixel classifier. For ViT the paper replaces the global query with the [CLS] query and adds the input back through the residual, so the ViT variant keeps the residual path ([arXiv PDF, Sec. 3](https://arxiv.org/pdf/2112.01071)). Two training-free refinements: key smoothing (average predictions over patches with similar key features) and prompt denoising (drop classes whose confidence stays below 0.5 everywhere).
- Numbers (annotation-free, Table 1a, 512x512 input): PASCAL Context 21.7 (ViT-B/16), 25.5 with key smoothing and prompt denoising; COCO-Stuff 12.5 and 14.6. The unmodified CLIP baseline scores 9.0 and 4.3. MaskCLIP+ (pseudo-label distillation into DeepLabv2-ResNet101) reaches 31.1 and 18.0.
- Resolution (Appendix E, Table 8, PASCAL Context): 224 px gives 22.72, 336 px gives 23.02, 520 px gives 21.68, multi-scale [224, 336, 520] gives 26.34. The authors name 336 the sweet spot.
- The paper evaluates only CLIP-RN50 and CLIP-ViT-B/16. The official repository ships configs for ViT-B/32 as well ([github.com/chongzhou96/MaskCLIP](https://github.com/chongzhou96/MaskCLIP)), but the paper reports no ViT-B/32 numbers. The authors attribute part of ViT's advantage over ResNet to the 16x versus 32x downsampling, which is directly relevant to a /32 patch grid.

### SCLIP (Wang, Mei and Yuille, ECCV 2024)
- Claim: replacing the last block's attention with correlative self-attention gives spatially covariant features. [arXiv 2312.01597](https://arxiv.org/abs/2312.01597)
- Method: attention = Softmax(X W_r W_r^T X^T / tau), reusing the pretrained W_q and W_k (ensemble), only in the last layer.
- Numbers: 38.2 average over eight benchmarks (table above). Input 336 short side, 224x224 windows with stride 112. Resolution ablation on VOC21: 224 short side 56.5, 336 short side 59.1, 448 short side 60.4. PAMR post-processing adds 1.9 average.
- No ViT-B/32 results in the SCLIP paper. NACLIP re-ran SCLIP on ViT-B/32 (below).

### ClearCLIP (Lan et al., ECCV 2024)
- Claim: the residual connection carries most of the noise in dense CLIP maps; removing it, using q-q attention and dropping the FFN of the last block gives clean maps. [arXiv 2407.12442](https://arxiv.org/abs/2407.12442)
- Numbers (448 short side, sliding window): VOC20 80.9, Context59 35.9, Stuff 23.9, Cityscapes 30.0, ADE20K 16.7 (average 37.5); VOC21 51.8, Context60 32.6, Object 33.0.
- Ablation (ViT-B/16, five datasets without background): q-q attention with residual and FFN gives 27.3 average; removing the residual alone gives 33.7; removing residual and FFN gives 37.5. The residual removal is the single largest gain. The paper also reports that the residual-stream norm relative to the attention-output norm differs between B/16 and L/14 and correlates with mIoU.
- Backbones: ViT-B/16 and ViT-L/14 for CLIP and OpenCLIP. No ViT-B/32 and no 224-px-without-window result.

### CLIP Surgery (Li et al., Pattern Recognition 2025; arXiv 2023)
- Claim: raw CLIP saliency is inverted (background lights up) and noisy; v-v attention in a parallel path plus "feature surgery" fixes both. [arXiv 2304.05653](https://arxiv.org/abs/2304.05653)
- Method: a new path from layer 7 onward that uses V V^T attention and no FFN, kept parallel to the original path; feature surgery subtracts the class-weighted mean ("redundant") feature shared across all text prompts from the image-text product.
- Numbers (ViT-B/16, open-vocabulary segmentation): PASCAL Context 29.3, COCO-Stuff 21.9, Cityscapes 31.4. Explainability metric mSC improves from 47.72 to 65.21 averaged over five backbones and four datasets. The paper evaluates RN50, RN101, ViT-B/32, ViT-B/16 and ViT-L/14; the per-backbone ViT-B/32 segmentation numbers were not extracted here and remain unverified.
- Relevance: feature surgery directly targets the "generic concept absorbs many patches" failure (floor, bed) seen in the local check.

### GEM (Bousselham et al., CVPR 2024)
- Claim: self-self attention (q-q, k-k, v-v ensemble) clusters tokens of the same object while keeping text alignment. [CVF open access](https://openaccess.thecvf.com/content/CVPR2024/html/Bousselham_Grounding_Everything_Emerging_Localization_Properties_in_Vision-Language_Transformers_CVPR_2024_paper.html), [arXiv 2312.00878](https://arxiv.org/abs/2312.00878)
- Method: parallel GEM path from the middle layers, L2-normalized projections, adaptive temperature, one iteration; 448 px short side with interpolated positional embeddings.
- Numbers (Table 5, VOC / PascalContext / OpenImages-V7 point mIoU):
  - ViT-B/16 CLIP 46.2 / 32.6 / 50.9; OpenCLIP 43.1 / 31.7 / 49.9; MetaCLIP 46.8 / 34.5 / 51.9.
  - **ViT-B/32 CLIP 40.5 / 27.0 / 46.6; ViT-B/32 OpenCLIP 39.3 / 23.9 / 45.5; ViT-B/32 MetaCLIP 38.2 / 28.2 / 46.7.**
  - The OpenCLIP pretraining set (for example laion2b) behind these rows was not verified.
  - Same-protocol baselines on ViT-B/16 VOC: CLIP 10.4, MaskCLIP 28.6, CLIP Surgery 41.2.

### NACLIP (Hajimiri, Ben Ayed and Dolz, WACV 2025)
- Claim: each patch should attend to its spatial neighbours; key-key similarity plus a Gaussian neighbourhood prior and a reduced last block give the best training-free CLIP-only results without auxiliary models. [arXiv 2404.08181](https://arxiv.org/abs/2404.08181), [code](https://github.com/sinahmr/NACLIP), [WACV 2025 record](https://mlanthology.org/wacv/2025/hajimiri2025wacv-pay/)
- Method: logits k k^T plus a 2D Gaussian (sigma = 5 patches); last block reduced to Z = SA(LN(Z_prev)), no FFN and no residual.
- Numbers (ViT-B/16, 336 short side, 224 window, stride 112): VOC21 58.9, PC60 32.2, COCO-Object 33.2, Cityscapes 35.5, ADE20K 17.4; 39.4 average over eight benchmarks without post-processing, 42.5 with PAMR.
- Ablation (VOC21 / PC60): vanilla CLIP 18.6 / 7.8; neighbourhood prior only 38.0 / 17.0; k-k only 36.0 / 15.6; both 40.2 / 17.4; plus reduced last block 58.9 / 32.2. Removing the residual and FFN is the largest single gain, matching ClearCLIP.
- **ViT-B/32 (Table 2, VOC21 / PC59 / average):** NACLIP 54.8 / 34.9 / 44.9; SCLIP 54.8 / 30.2 / 42.5; GEM 40.5 / 27.0 / 33.8. ViT-B/16 in the same table: NACLIP 64.1 / 38.4 / 51.3. The ViT-B/16 VOC21 entry (64.1) differs from the main-table value (58.9); the paper does not explain which protocol Table 2 uses, so treat Table 2 as an internal backbone comparison only.

### ResCLIP (Yang et al., CVPR 2025)
- Claim: intermediate-layer cross-correlation attention and a semantic feedback step add about 2 mIoU on top of existing training-free methods. [arXiv 2411.15851](https://arxiv.org/abs/2411.15851), [CVF](https://openaccess.thecvf.com/content/CVPR2025/papers/Yang_ResCLIP_Residual_Attention_for_Training-free_Dense_Vision-language_Inference_CVPR_2025_paper.pdf)
- Numbers (ViT-B/16, five datasets without background): SCLIP 37.1 to 39.3, ClearCLIP 37.5 to 40.0, NACLIP 38.2 to 40.3. No ViT-B/32 result found.

### What matters, according to the ablations
1. Removing the residual connection and the FFN in the last block gives the largest single improvement (ClearCLIP 27.3 to 37.5; NACLIP about +12 average). The MaskCLIP ViT variant as written in the paper keeps the residual.
2. Replacing q-k attention with a self-similarity (q-q, k-k, v-v or W W^T) is the second ingredient; a locality prior (NACLIP) helps further.
3. Resolution: 336 px beats 224 px for MaskCLIP (23.02 vs 22.72) and SCLIP (59.1 vs 56.5 VOC21); multi-scale ensembling helps MaskCLIP most (26.34). All recent methods use a 224 window slid over a 336 or 448 image, which for ViT-B/32 means several 7x7 grids stitched into a finer map.
4. ViT-B/32 costs roughly 6 to 10 mIoU against ViT-B/16 for the same method (NACLIP 51.3 vs 44.9; GEM 46.2 vs 40.5 VOC). OpenCLIP ViT-B/32 sits slightly below OpenAI ViT-B/32 under GEM (39.3 vs 40.5 VOC).

---

## 2. Language- or saliency-guided attention against spurious correlations

### GALS (Petryk et al., CVPR 2022)
- Claim: language specification of the class, grounded by CLIP attention, tells a CNN which pixels matter. [arXiv 2202.08926](https://arxiv.org/abs/2202.08926), [CVF](https://openaccess.thecvf.com/content/CVPR2022/html/Petryk_On_Guiding_Visual_Attention_With_Language_Specification_CVPR_2022_paper.html), [code](https://github.com/spetryk/GALS)
- Method: GradCAM of the CLIP image-text score ("an image of a bird", "a photo of a bird") on CLIP-RN50 gives an attention map; an RRR-style L1 penalty on the input gradients of an ImageNet-pretrained ResNet-50 where the CLIP map is low.
- Supervision: class labels plus a short text description per class. No group labels, also not for validation (model selection on class accuracy).
- Numbers (Table 10, test, per-group mean / worst-group):
  - Waterbirds-95%: Vanilla 86.93 / 73.07; UpWeight 86.74 / 73.66; ABN 86.01 / 65.03; GALS 89.05 / 76.54. Zero-shot CLIP 73.18 / 43.46.
  - Waterbirds-100% (perfect correlation): Vanilla 69.83 / 34.31; ABN 72.20 / 41.56; GALS 79.72 / 56.71; logistic regression on CLIP features 68.36 / 32.15.
  - Red Meat (Food-101 subset) accuracy: GALS 71.20, Vanilla 67.39, ABN 69.44.
- Ablation: CLIP ViT-B/32 was tried as the attention source (Table 3); CLIP-RN50 GradCAM with RRR was chosen as the most consistent.
- Limitation stated by the authors: the method applies only to biases that are pixel-wise separable from the relevant features.

### Right for the Right Reasons (Ross, Hughes and Doshi-Velez, IJCAI 2017)
- Claim: penalizing input gradients inside an annotation mask of irrelevant features makes models use the right features and generalize under train-test shift. [arXiv 1703.03717](https://arxiv.org/abs/1703.03717)
- Supervision: per-example (or per-feature) expert annotation masks. GALS replaces the expert mask with a CLIP attention map. No Waterbirds numbers in the original paper.

### Guiding models with explanations (Rao et al., ICCV 2023)
- Claim: an "Energy" localization loss (negative fraction of positive attribution inside the box) guides models toward object features even with coarse bounding boxes and only 1% annotated images. [arXiv 2303.11932](https://arxiv.org/abs/2303.11932)
- Numbers on Waterbirds-100 with 1% box annotations (B-cos models): WGA 43.4 baseline, 56.1 Energy, 51.1 L1. Reversed task (classify background): 56.6, 62.8, 58.8. These values come from the HTML version and were not cross-checked in the PDF.

### MaskTune (Asgari et al., NeurIPS 2022)
- Claim: masking the features an ERM model relies on (from its saliency) and fine-tuning one epoch forces it to find new features, with no group labels at all. [arXiv 2210.00055](https://arxiv.org/abs/2210.00055)
- Numbers (ImageNet-pretrained ResNet-50): CelebA WGA 78.0 (ERM 47.2, JTT without group-labelled validation 40.6, GroupDRO 88.3). Waterbirds WGA 86.4 against ERM 80.8 and GroupDRO 89.3, measured on the authors' corrected Waterbirds; these Waterbirds numbers are not comparable to the standard split.

### DFR (Kirichenko, Izmailov and Wilson, ICLR 2023)
- Claim: ERM models on spurious benchmarks still learn the core features; retraining only the last layer on a small group-balanced set recovers high WGA. [arXiv 2204.02937](https://arxiv.org/abs/2204.02937), [code](https://github.com/PolinaKirichenko/deep_feature_reweighting)
- Evidence: an ImageNet-pretrained ResNet-50 trained on Waterbirds-100% scores 38.4 WGA on original images but about 94 on foreground-only test images (Table 1). On Dominoes with 99% or 95% correlation, a logistic probe on the frozen features nearly matches the optimal core-only accuracy; with 100% correlation the probe still recovers the core feature for MNIST-MNIST and MNIST-Fashion, but not for MNIST-CIFAR where the complexity gap is largest (Sec. 4.2, Fig. 2).
- Numbers (Table 2, WGA / mean): Waterbirds DFR 92.9 / 94.2, ERM 74.9 / 98.1, GroupDRO 91.4 / 93.5; CelebA DFR 88.3 / 91.3, ERM 46.9 / 95.3, GroupDRO 88.9 / 92.9.
- Relevance: the thesis probe is DFR applied to SSL features. DFR works when the encoder already represents the core feature; the thesis failure mode (the student represents only the shared component) is exactly the case where DFR has nothing to reweight. DFR's own boundary condition is perfect correlation combined with a large simplicity gap.

---

## 3. Background augmentation and counterfactual views in SSL

### Ryali, Schwab and Morcos 2021: background augmentations for SSL
- Claim: removing or swapping backgrounds with a saliency mask makes SSL focus on the foreground and improves accuracy and robustness. [arXiv 2103.12719](https://arxiv.org/abs/2103.12719) (arXiv; no venue verified)
- Augmentations: BG_RM (grey background), BG_Random (another image's inpainted background) and BG_Swaps (query and positive share the foreground with different backgrounds; a negative shares the query's background, built as m_q r + (1 - m_q) q). Masks from DeepUSPS2, an unsupervised saliency detector; substantial mask noise is tolerated (Appendix B). Applied with probability 0.1 to 0.3; always-on hurts.
- Numbers: ImageNet linear accuracy +0.6 to 1.4 for MoCo-v2, BYOL and SwAV (Table 2). ImageNet-9 (Table 10), MoCo-v2 baseline / BG_RM / BG_Swaps: Only-FG 74.4 / 81.9 / 86.1; Mixed-Rand 70.7 / 76.3 / 84.1; Mixed-Next 67.0 / 73.0 / 82.2. BYOL Mixed-Rand 79.6 to 85.5 with BG_Random. The BG-gap shrinks for every method.
- Ablation (Table 3 and 4): BG_RM applied only to the query gives no gain; it needs matched negatives. BG_Random still helps when applied only to the query, more when also in the positive. Each BG_Swaps component adds and the gains stack (MoCo-v2 68.3 to 69.7 in the ablation setting).

### Xiao et al., ICLR 2021: Noise or Signal
- Claim: ImageNet models use background signal; backgrounds alone give 40 to 50% accuracy on 9 classes; adversarial backgrounds flip 87.5% of foregrounds. [OpenReview](https://openreview.net/forum?id=gl3D-xY7wLq), [arXiv 2006.09994](https://arxiv.org/abs/2006.09994), [code](https://github.com/MadryLab/backgrounds_challenge)
- Dataset: ImageNet-9 with Only-FG, Mixed-Same, Mixed-Rand, Mixed-Next and background-only variants. BG-gap (Mixed-Same minus Mixed-Rand) is 13 to 22% for IN-9L-trained models and 6 to 14% for ImageNet-trained models.
- Intervention: training on Mixed-Rand (random backgrounds) raises Mixed-Rand accuracy by 17.3 points and Mixed-Next by 22.3 points; it also drops background-only accuracy to about 15% (near chance 11%).

### ContrastiveCrop (Peng et al., CVPR 2022)
- Claim: localizing the object from the model's own feature heatmap and constraining crop centres inside its box, plus center-suppressed sampling, gives better positives. [arXiv 2202.03278](https://arxiv.org/abs/2202.03278), [code](https://github.com/xyupeng/ContrastiveCrop)
- Setting close to the thesis: ResNet-18, 500 epochs, batch 512. Linear accuracy RandomCrop vs ContrastiveCrop for SimCLR: CIFAR-10 89.63 vs 90.08, CIFAR-100 60.30 vs 61.91, Tiny ImageNet 45.19 vs 46.21, STL-10 88.95 vs 89.53. Gains 0.4 to 2.0 across SimCLR, MoCo, BYOL and SimSiam. No WGA reported.

### CAST (Selvaraju et al., CVPR 2021)
- Claim: saliency-constrained crops plus a Grad-CAM attention loss (query Grad-CAM pushed onto the salient region) fix poor grounding of contrastive SSL on scene images. [arXiv 2012.04630](https://arxiv.org/abs/2012.04630)
- Numbers (MoCo on COCO, Table 1): VOC07 linear mAP 66.9 to 73.1; VOC detection AP 47.5 to 54.2. Backgrounds Challenge (Table 3), MoCo / CAST: Original 72.62 / 77.33, Mixed-Same 45.75 / 54.42, Mixed-Rand 30.44 / 39.93, Mixed-Next 26.86 / 37.46, Only-FG 30.42 / 43.26; background-only variants drop slightly, the intended direction.
- Saliency from DeepUSPS (unsupervised). Constrained crop alone gives part of the gain; the Grad-CAM loss gives the rest.

### DiLo (Zhao et al., AAAI 2021)
- Claim: copy-pasting saliency-estimated foregrounds onto other backgrounds teaches instance discrimination to ignore backgrounds. [arXiv 2004.06638](https://arxiv.org/abs/2004.06638)
- Numbers (ImageNet linear, 200 epochs, ResNet-50, Table 2): MoCo 60.6; with RBD saliency (hand-crafted, no learning) 62.8; with BASNet 65.0. Copy-paste probability 30% gives 62.8, 50% 62.2, 70% 61.6, 100% 47.6. Grey and ImageNet backgrounds work equally (62.8 and 62.1); texture backgrounds give nothing. Blending adds about 0.4. Table 3: MoCo-v2 67.5 to 69.2 with BASNet.
- Negative result directly relevant to option A: masking (pooling) the final-layer features with the saliency map drops accuracy by 19%, attributed to lost context.

### Simple Copy-Paste (Ghiasi et al., CVPR 2021)
- Supervised instance segmentation: random pasting of masked objects with large-scale jitter, no blending, no context modelling; COCO 49.1 mask AP and 57.3 box AP. [arXiv 2012.07177](https://arxiv.org/abs/2012.07177). It supports the design choice that random placement without context reasoning suffices.

### SSL evaluated on Waterbirds or MetaShift WGA
- **LateTVG (Hamidieh, Zhang and Ghassemi, ICLR 2024)** is the closest prior work to the thesis setting. [ICLR proceedings PDF](https://proceedings.iclr.cc/paper_files/paper/2024/file/9c7900fac04a701cbed83256b76dbaa3-Paper-Conference.pdf), [arXiv 2406.18562](https://arxiv.org/abs/2406.18562)
  - Setting: SimSiam or SimCLR, ResNet backbone initialized randomly, pretraining on the spuriously correlated train split, logistic probe on a group-balanced downsampled set, test WGA. The exact ResNet depth and epochs were not found in the main paper.
  - Method: the second view is passed through a copy of the encoder with magnitude pruning of the later layers ("feature-space augmentation"). Re-sampling the pretraining data does not reliably help (Table 2). The paper argues that standard augmentations change class or spurious attribute more often than neither ("spurious connectivity").
  - WGA (Table 3), SSL-Base to LateTVG:
    - SimCLR: CelebA 76.7 to 82.2; CMNIST 81.7 to 83.8; **MetaShift 45.5 to 59.3**; **Spur-CIFAR10 36.5 to 40.4**; **Waterbirds 43.8 to 55.4**.
    - SimSiam: CelebA 77.5 to 83.1; CMNIST 80.7 to 83.1; MetaShift 42.3 to 79.6; Spur-CIFAR10 43.4 to 61.4; Waterbirds 48.3 to 56.3.
    - The table layout in the PDF text extraction is ambiguous; the row assignment above follows the row order and the paper's remark that CMNIST shows the smallest gain. Check the rendered PDF before citing a single number.
  - The paper reports the best hyperparameter combination per dataset.
- **LA-SSL (Zhu et al., arXiv 2023)**: sample training images inversely to their learning speed during SSL; evaluated on Corrupted CIFAR-10, CelebA and MIMIC-CXR with precision/recall per subgroup, no Waterbirds or MetaShift WGA. [arXiv 2311.16361](https://arxiv.org/abs/2311.16361)
- **GRSSL (Yadav et al., CVPR 2026 workshop)**: Grounding-DINO plus SAM foregrounds, FLUX.1-Fill background inpainting, cross-variant contrastive positives, then GroupDRO on an ImageNet-pretrained ResNet-50. WGA Waterbirds 92.5, MetaShift 81.7; without the cross-variant SSL term Waterbirds drops to 89.6. It uses group labels and a pretrained backbone, so it is a counterfactual-view reference only. [arXiv 2607.05850](https://arxiv.org/abs/2607.05850). Numbers from the HTML version only.

---

## 4. Dense or spatial distillation and region-level SSL

### DenseCL (Wang et al., CVPR 2021, oral)
- Pixel-level contrastive loss between two views, with correspondences from backbone feature similarity, added to the global MoCo-v2 loss. Gains over MoCo-v2: VOC detection +2.0 AP, COCO detection +1.1 AP, COCO instance segmentation +0.9 AP, VOC segmentation +3.0 mIoU, Cityscapes +1.8 mIoU; under 1% slower. [arXiv 2011.09157](https://arxiv.org/abs/2011.09157)

### PixPro (Xie et al., CVPR 2021)
- Pixel-to-propagation consistency: positives are pixels close in the original image coordinates across the two crops; a propagation module smooths features; combined with an instance-level loss. ResNet-50: VOC detection (C4) 60.2 AP (+2.6), COCO 41.4 / 40.5 mAP (FPN / C4), Cityscapes 77.2 mIoU. [arXiv 2011.10043](https://arxiv.org/abs/2011.10043), [code](https://github.com/zdaxie/PixPro)
- Relevance: the "known geometric correspondence across crops" trick is exactly what makes per-location targets well defined under SimCLR crops; teacher maps must be warped by the same crop and flip.

### DetCon (Hénaff et al., ICCV 2021)
- Contrastive loss over mask-pooled features of the ResNet-50 7x7 map, with masks from a grid, Felzenszwalb-Huttenlocher, MCG or ground truth; up to 10x less pretraining than SimCLR or BYOL for equal transfer. [arXiv 2103.10957](https://arxiv.org/abs/2103.10957)
- The paper reports that better masks give better transfer, with ground-truth masks best. The per-mask COCO AP values were read only through an automated summary and are not verified here.
- Relevance: DetCon is the closest template for option A at 7x7: pool the student's 7x7 map under each concept mask and contrast or regress those region vectors.

### Attention transfer (Zagoruyko and Komodakis, ICLR 2017)
- Spatial attention map F(A) = sum over channels of |A_c|^p; loss = distance between L2-normalized vectorized student and teacher maps. CIFAR-10 WRN-16-1 student with WRN-16-2 teacher: error 8.77 to 7.93 (AT) and 6.31 (AT + KD). [arXiv 1612.03928](https://arxiv.org/abs/1612.03928)
- Relevance: a channel-free loss that only constrains where the student is active, which avoids having to match the teacher's feature space.

### Cross-architecture distillation (OFA-KD, Hao et al., NeurIPS 2023)
- CKA analysis shows ViT and CNN intermediate features diverge strongly, so hint-based (feature-map) distillation between them is ineffective; projecting features into the logit space fixes it. Gains up to 8.0 on CIFAR-100 and 0.7 on ImageNet-1K. [arXiv 2310.19444](https://arxiv.org/abs/2310.19444), [NeurIPS 2023](https://proceedings.neurips.cc/paper_files/paper/2023/file/fb8e5f198c7a5dcd48860354e38c0edc-Paper-Conference.pdf)
- Relevance: distil CLIP concept maps through a small head into a text-score (logit) space.

### CLIP-KD (Yang et al., CVPR 2024)
- Compares relation, feature, gradient and contrastive distillation of CLIP; plain feature mimicry with MSE works best, interactive contrastive learning also works. Students ViT-B/16 and ResNet-50 reach 57.5 and 55.4 zero-shot ImageNet top-1 when trained on CC3M+12M with a ViT-L/14 teacher. [arXiv 2307.12732](https://arxiv.org/abs/2307.12732), [CVF](https://openaccess.thecvf.com/content/CVPR2024/papers/Yang_CLIP-KD_An_Empirical_Study_of_CLIP_Model_Distillation_CVPR_2024_paper.pdf)

### AM-RADIO (Ranzinger et al., CVPR 2024)
- Multi-teacher distillation of CLIP, DINOv2 and SAM into a student trained from scratch, with both summary and spatial losses. Spatial loss = 0.9 cosine + 0.1 smooth-L1, student and teacher maps bilinearly interpolated to the larger resolution, one 2-layer MLP adaptor per teacher. [arXiv 2312.06709](https://arxiv.org/abs/2312.06709)
- Findings: adding the spatial feature loss always helps; DINOv2 provides better spatial features than CLIP, both together are best. CNN students (Table 7) trail ViTs on ADE20k: EfficientNetV2-S 27.75, ResNetv2-101 29.61, ConvNeXt-B 38.95 (teacher DINOv2-G 47.53).
- This is the only work found that distils dense CLIP features into a student from scratch; it uses web-scale data. MaskCLIP+ distils dense CLIP pseudo-labels into an ImageNet-initialized DeepLab. A search for dense CLIP distillation into a from-scratch CNN on a small spurious-correlation dataset found no prior work; the thesis would be the first such test.

---

## 5. Concept decoupling, erasure and identifiability

### LEACE (Belrose et al., NeurIPS 2023)
- Claim: a closed-form eraser that makes a concept Z linearly unpredictable while changing the representation as little as possible in every inner-product norm. [arXiv 2306.03819](https://arxiv.org/abs/2306.03819), [code](https://github.com/EleutherAI/concept-erasure)
- Key theorem: X linearly guards Z (no linear predictor beats the constant predictor for any convex loss) if and only if the class-conditional means of X equal the unconditional mean, equivalently Cov(X, Z) = 0.

### INLP (Ravfogel et al., ACL 2020)
- Repeatedly fit a linear classifier for the protected attribute and project the representation onto its nullspace until the attribute is linearly unpredictable. [arXiv 2004.07667](https://arxiv.org/abs/2004.07667)

### RLACE (Ravfogel et al., ICML 2022)
- Linear minimax game over rank-k orthogonal projections, solved through a convex relaxation; removes a concept with fewer dimensions than INLP. [arXiv 2201.12091](https://arxiv.org/abs/2201.12091)

### Why erasing B and predicting A equals predicting A's residual on B
Let r(X) be any representation with Cov(r(X), B) = 0 (the LEACE guardedness condition). Write A = gamma B + A_perp with gamma = Cov(A, B) Cov(B)^-1, so A_perp is the residual of A after linear regression on B and Cov(A_perp, B) = 0. Then Cov(r(X), A) = gamma Cov(r(X), B) + Cov(r(X), A_perp) = Cov(r(X), A_perp). The least-squares coefficients of A on r(X), Cov(r(X))^+ Cov(r(X), A), are therefore identical to those of A_perp on r(X): the two problems have the same solution and the same predictions. The explainable share of Var(A) is at most Var(A_perp) / Var(A) = 1 - rho_AB^2, because the gamma B part is by construction uncorrelated with r(X). This is the Frisch-Waugh-Lovell theorem of linear regression ([Frisch and Waugh 1933](https://doi.org/10.2307/1907330), [Lovell 1963](https://doi.org/10.1080/01621459.1963.10480682)) applied to the erased representation. When "cat" and "couch" correlate at image level with rho close to 1, the residual target has little variance and the student can only learn whatever image-level signal separates them; erasure adds no information beyond the residual target that already failed. (Derivation by the author of this note; the guardedness condition is LEACE's theorem.)

### Identifiability: why image-level statistics cannot separate correlated factors
- **Locatello et al., ICML 2019**: unsupervised learning of disentangled representations is impossible without inductive biases on both models and data; 12,000 trained models confirm that supervision is needed to select disentangled models. [arXiv 1811.12359](https://arxiv.org/abs/1811.12359)
- **Locatello et al., ICML 2020**: pairs of observations that share some factors (without knowing which or how many) suffice for identifiable disentanglement. [arXiv 2002.02886](https://arxiv.org/abs/2002.02886)
- **Träuble et al., ICML 2021**: correlations between factors in the training data persist in the learned latents (4,260 models); weak supervision during training or post-hoc correction with a few labels resolves them. [arXiv 2006.07886](https://arxiv.org/abs/2006.07886)
- **von Kügelgen et al., NeurIPS 2021**: contrastive SSL identifies the content block (the latent part that augmentations leave unchanged) up to an invertible map, even when content and style are statistically or causally dependent. [arXiv 2106.04619](https://arxiv.org/abs/2106.04619)
- Synthesis: the pair (cat, couch) is separable only through structure that varies one factor while holding the other: paired observations (Locatello 2020), augmentations that change one block (von Kügelgen 2021) or spatial structure, where cat and couch occupy different pixels even when they co-occur in every image. The spatial direction supplies exactly this structure: within an image, the location index acts as the intervention that image-level targets lack. Counterfactual cut-paste views turn that spatial separation into the augmentation-defined content/style split that von Kügelgen et al. prove identifiable.

---

## Implications for the spatial method

1. **Localizer at 7x7: use NACLIP-style or ClearCLIP-style last-block surgery over plain MaskCLIP.** Drop the residual and FFN of the last block and use self-similarity attention (k-k with a Gaussian neighbourhood prior). This is the largest ablated gain in both ClearCLIP (27.3 to 37.5) and NACLIP (about +12). NACLIP is also the only method with a published ViT-B/32 row (VOC21 54.8, PC59 34.9, against SCLIP 54.8 / 30.2 and GEM 40.5 / 27.0). Check whether the current local "value path" keeps the residual: the MaskCLIP ViT formulation keeps it. Sources: [ClearCLIP](https://arxiv.org/abs/2407.12442), [NACLIP](https://arxiv.org/abs/2404.08181), [MaskCLIP](https://arxiv.org/pdf/2112.01071).
2. **Compute teacher maps at higher resolution offline, then pool to 7x7.** Every method gains from 336 to 448 px with 224 sliding windows (SCLIP VOC21 56.5 at 224 vs 59.1 at 336 vs 60.4 at 448; MaskCLIP multi-scale 26.34 vs 22.72). With ViT-B/32, a 448 image gives a 14x14 map; average-pool it to the student's 7x7 grid. The cost is one cached pass per training image. Sources: [SCLIP](https://arxiv.org/abs/2312.01597), [MaskCLIP](https://arxiv.org/pdf/2112.01071).
3. **Suppress generic concepts before using the maps.** Apply CLIP Surgery's feature surgery (subtract the class-averaged redundant feature) or MaskCLIP's prompt denoising. Then score each concept against the full concept vocabulary with a softmax across concepts per patch, so "floor" and "bed" stop absorbing patches. Sources: [CLIP Surgery](https://arxiv.org/abs/2304.05653), [MaskCLIP](https://arxiv.org/pdf/2112.01071).
4. **Option A loss: distil into a logit-like concept space through a small head, with cosine loss and crop-aware warping.** Cross-architecture feature-map matching between ViT and CNN fails without a projection ([OFA-KD](https://arxiv.org/abs/2310.19444)); AM-RADIO's spatial loss (0.9 cosine + 0.1 smooth-L1, per-teacher MLP adaptor, bilinear resize) is the tested recipe ([AM-RADIO](https://arxiv.org/abs/2312.06709)). Two cheaper alternatives exist. Attention transfer constrains only where the student is active ([AT](https://arxiv.org/abs/1612.03928)). DetCon-style mask pooling contrasts region vectors per concept mask ([DetCon](https://arxiv.org/abs/2103.10957)). Warp teacher maps with the exact SimCLR crop and flip parameters, as PixPro does with pixel correspondences ([PixPro](https://arxiv.org/abs/2011.10043)). Avoid hard feature masking at the pooling stage: DiLo reports a 19% drop from saliency-masked pooling ([DiLo](https://arxiv.org/abs/2004.06638)).
5. **Option B design: background swaps with matched negatives, applied with probability 0.1 to 0.5.** BG_Swaps (positive with a new background, negative sharing the query's background) beat BG_RM for MoCo-v2 on ImageNet-9 Mixed-Rand (baseline 70.7, BG_RM 76.3, BG_Swaps 84.1) ([Ryali et al.](https://arxiv.org/abs/2103.12719)). DiLo shows that always-on copy-paste collapses accuracy (47.6) while 30% helps; noisy hand-crafted saliency already works ([DiLo](https://arxiv.org/abs/2004.06638)). Suggested SimCLR translation (this note's proposal): paste each foreground onto another batch image's background and keep that background image in the batch, so it serves as the matched-background negative. Simple random placement without blending suffices ([Copy-Paste](https://arxiv.org/abs/2012.07177)).
6. **Add a localization prior on crops as a cheap first step.** CAST's saliency-constrained crops and ContrastiveCrop's heatmap-constrained crops both help SimCLR-family methods; ContrastiveCrop uses ResNet-18, 500 epochs, which matches the thesis setup ([ContrastiveCrop](https://arxiv.org/abs/2202.03278), [CAST](https://arxiv.org/abs/2012.04630)). The CLIP concept map can replace their saliency source.
7. **Baselines the thesis must report.** (a) SimCLR baseline with the same balanced probe; (b) LateTVG, the published from-scratch SSL method on MetaShift, Waterbirds, Spur-CIFAR10 and CelebA with a balanced probe (SimCLR WGA MetaShift 45.5 to 59.3, Waterbirds 43.8 to 55.4) ([LateTVG](https://arxiv.org/abs/2406.18562)); (c) BG_Random or BG_Swaps with a non-CLIP saliency source, to separate "any foreground mask" from "concept-specific CLIP mask" ([Ryali et al.](https://arxiv.org/abs/2103.12719)); (d) GALS-style input-gradient guidance as the supervised-attention reference ([GALS](https://arxiv.org/abs/2202.08926)); (e) the frozen CLIP ViT-B/32 probe and the existing image-level distillation arms as upper and negative references; (f) DFR numbers on ImageNet-pretrained ResNet-50 (Waterbirds 92.9, CelebA 88.3) as the ceiling for features that already contain the core attribute ([DFR](https://arxiv.org/abs/2204.02937)).
8. **Frame the result with the identifiability literature.** The image-level failure matches Träuble et al. (correlations persist in learned latents) and the Frisch-Waugh-Lovell argument above; the spatial method is the test of whether within-image location supplies the factor-varying structure that Locatello et al. 2020 and von Kügelgen et al. 2021 require ([Träuble et al.](https://arxiv.org/abs/2006.07886), [Locatello 2020](https://arxiv.org/abs/2002.02886), [von Kügelgen 2021](https://arxiv.org/abs/2106.04619)).
