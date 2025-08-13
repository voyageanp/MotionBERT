Abstract
We present a unified perspective on tackling various
human-centric video tasks by learning human motion representations from large-scale and heterogeneous data resources. Specifically, we propose a pretraining stage in
which a motion encoder is trained to recover the underlying 3D motion from noisy partial 2D observations. The
motion representations acquired in this way incorporate
geometric, kinematic, and physical knowledge about human motion, which can be easily transferred to multiple
downstream tasks. We implement the motion encoder with
a Dual-stream Spatio-temporal Transformer (DSTformer)
neural network. It could capture long-range spatio-temporal
relationships among the skeletal joints comprehensively and
adaptively, exemplified by the lowest 3D pose estimation
error so far when trained from scratch. Furthermore, our
proposed framework achieves state-of-the-art performance
on all three downstream tasks by simply finetuning the pretrained motion encoder with a simple regression head (1-2
layers), which demonstrates the versatility of the learned
motion representations. Code and models are available at
https://motionbert.github.io

1. Introduction
Perceiving and understanding human activities have long
been a core pursuit of machine intelligence. To this end,
researchers define various tasks to estimate human-centric
semantic labels from videos, e.g. skeleton keypoints [14,35],
action classes [64, 123], and surface meshes [46, 71]. While
significant progress has been made in each of these tasks,
they tend to be modeled in isolation, rather than as interconnected problems. For example, Spatial Temporal Graph Convolutional Networks (ST-GCN) have been applied to modeling spatio-temporal relationship of human joints in both 3D
pose estimation [13, 116] and action recognition [96, 123],
but their connections have not been fully explored. Intuitively, these models should all have learned to identify typical human motion patterns, despite being designed for different problems. Nonetheless, current methods fail to mine
and utilize such commonalities across the tasks. Ideally, we
could develop a unified human-centric video representation
that can be shared across all relevant tasks.
One significant challenge to developing such a representation is the heterogeneity of available data resources. Motion capture (Mocap) systems [38, 76] provide high-fidelity
3D motion data obtained with markers and sensors, but the
appearances of captured videos are usually constrained to
simple indoor scenes. Action recognition datasets provide
annotations of the action semantics, but they either contain
no human pose labels [16, 95] or feature limited motion
of daily activities [63, 64, 93]. In contrast, in-the-wild human videos offer a vast and diverse range of appearance and
motion. However, obtaining precise 2D pose annotations
requires considerable effort [3], and acquiring ground-truth
(GT) 3D joint locations is almost impossible. Consequently
most existing studies focus on a specific task using a single
type of human motion data, and they are not able to enjoy
the advantages of other data resources.
In this work, we provide a new perspective on learning
human motion representations. The key idea is that we can
learn a versatile human motion representation from heterogeneous data resources in a unified manner, and utilize the
representation to handle different downstream tasks in a
unified way. We present a two-stage framework, consisting of pretraining and finetuning, as depicted in Figure 1.
In the pretraining stage, we extract 2D skeleton sequences
from diverse motion data sources and corrupt them with random masks and noises. Subsequently, we train the motion
encoder to recover the 3D motion from the corrupted 2D
skeletons. This challenging pretext task intrinsically requires
the motion encoder to i) infer the underlying 3D human structures from its temporal movements; ii) recover the erroneous
and missing observations. In this way, the motion encoder
implicitly captures human motion commonsense such as
joint linkages, anatomical constraints, and temporal dynamics. In practice, we propose Dual-stream Spatio-temporal
Transformer (DSTformer) as the motion encoder to capture
the long-range relationship among skeleton keypoints. We
suppose that the motion representations learned from largescale and diversified data resources could be shared across
different downstream tasks and benefit their performance.
Therefore, for each downstream task, we adapt the pretrained
motion representations using task-specific training data and
supervisory signals with a simple regression head.
In summary, the contributions of this work are three-fold:
1) We provide a new perspective on solving various humancentric video tasks through a shared framework of learning
human motion representations. 2) We propose a pretraining
method to leverage the large-scale yet heterogeneous human
motion resources and learn generalizable human motion
representations. Our approach could take advantage of the
precision of 3D mocap data and the diversity of in-the-wild
RGB videos at the same time. 3) We design a dual-stream
Transformer network with cascaded spatio-temporal selfattention blocks that could serve as a general backbone for
human motion modeling. The experiments demonstrate that
the above designs enable a versatile human motion representation that can be transferred to multiple downstream tasks,
outperforming the task-specific state-of-the-art methods.

2. Related Work
Learning Human Motion Representations. Early works
formulate human motion with Hidden Markov Models [53,
108] and graphical models [51, 99]. Kanazawa et al. [42]
design a temporal encoder and a hallucinator to learn representations of 3D human dynamics. Zhang et al. [132]
predict future 3D dynamics in a self-supervised manner.
Sun et al. [102] further incorporate action labels with an
action memory bank. From the action recognition perspective, a variety of pretext tasks are designed to learn motion representations in a self-supervised manner, including future prediction [100], jigsaw puzzle [60], skeletoncontrastive [107], speed change [101], cross-view consistency [62], and contrast-reconstruction [117]. Similar
techniques are also explored in tasks like motion assessment [33,85] and motion retargeting [126,139]. These methods leverage homogeneous motion data, design corresponding pretext tasks, and apply them to a specific downstream
task. In this work, we propose a unified pretrain-finetune
framework to incorporate heterogeneous data resources and
demonstrate its versatility in various downstream tasks.

3D Human Pose Estimation. Recovering 3D human
poses from monocular RGB videos is a classical problem,
and the methods can be categorized into two categories.
The first is to estimate 3D poses with CNN directly from
images [82, 104, 136]. However, one limitation of these
approaches is that there is a trade-off between 3D pose
precision and appearance diversity due to current data collection techniques. The second category is to extract the
2D pose first, then lift the estimated 2D pose to 3D with
a separate neural network. The lifting can be achieved via
Fully Connected Network [29, 78], Temporal Convolutional
Network (TCN) [22, 89], GCN [13, 28, 116], and Transformer [56, 94, 134, 135]. Our framework is built upon the
second category as we use the proposed DSTformer to accomplish 2D-to-3D lifting.

Skeleton-based Action Recognition. The pioneering
works [74, 115, 127] point out the inherent connection between action recognition and human pose estimation. Towards modeling the spatio-temporal relationship among human joints, previous studies mainly employ LSTM [98, 138]
and GCN [21, 55, 68, 96, 123]. Most recently, PoseConv3D [32] proposes to apply 3D-CNN on the stacked 2D
joint heatmaps and achieves improved results. In addition to
the fully-supervised action recognition task, NTU-RGB+D120 [64] brings attention to the challenging one-shot action
recognition problem. To this end, SL-DML [81] applies deep
metric learning to multi-modal signals. Sabater et al. [92]
explores one-shot recognition in therapy scenarios with TCN.
We demonstrate that the pretrained motion representations
could generalize well to action recognition tasks, and the
pretrain-finetune framework is a suitable solution for the
one-shot challenges.

Human Mesh Recovery. Based on the parametric human
models such as SMPL [71], many research works [41,75,83,
122, 133] focus on regressing the human mesh from a single
image. SPIN [48] additionally incorporates fitting the body
model to 2D joints in the training loop. Despite their promising per-frame results, these methods yield jittery and unstable results [46,130] when applied to videos. To improve their
temporal coherence, PoseBERT [8] and SmoothNet [130]
propose to employ a denoising and smoothing module to the
single-frame predictions. Several works [24,42,46,106] take
video clips as input to exploit the temporal cues. Another
common problem is that paired images and GT meshes are
mostly captured in constrained scenarios, which limits the
generalization ability of the above methods. To that end,
Pose2Mesh [25] proposes to first extract 2D skeletons using
an off-the-shelf pose estimator, then lift them to 3D mesh
vertices. Our approach is complementary to state-of-the-art
human mesh recovery methods and could further improve
their temporal coherence with the pretrained motion representations.


Figure 2. Model architecture. We propose the Dual-stream Spatio-temporal Transformer (DSTformer) as a general backbone for human
motion modeling. DSTformer consists of N dual-stream-fusion modules. Each module contains two branches of spatial or temporal MHSA
and MLP. The Spatial MHSA models the connection among different joints within a timestep, while the Temporal MHSA models the
movement of one joint.

3. Method
3.1. Overview
As discussed in Section 1, our approach consists of two
stages, namely unified pretraining and task-specific finetuning. In the first stage, we train a motion encoder to
accomplish the 2D-to-3D lifting task, where we use the proposed DSTformer as the backbone. In the second stage,
we finetune the pretrained motion encoder and a few new
layers on the downstream tasks. We use 2D skeleton sequences as input for both pretraining and finetuning because
they could be reliably extracted from all kinds of motion
sources [3, 10, 76, 86, 103], and is more robust to variations [19, 32]. Existing studies have shown the effectiveness
of using 2D skeleton sequences for different downstream
tasks [25, 32, 89, 109]. We will first introduce the architecture of DSTformer, and then describe the training scheme in
detail.
3.2. Network Architecture
Figure 2 shows the network architecture for 2D-to-3D
lifting. Given an input 2D skeleton sequence x ∈ R
T ×J×Cin
,
we first project it to a high-dimensional feature F
0 ∈
R
T ×J×Cf
, then add learnable spatial positional encoding
PS
pos ∈ R
1×J×Cf and temporal positional encoding PT
pos ∈
R
T ×1×Cf
to it. We then use the sequence-to-sequence model
DSTformer to calculate F
i ∈ R
T ×J×Cf
(i = 1, . . . , N)
where N is the network depth. We apply a linear layer with
tanh activation [30] to F
N to compute the motion representation E ∈ R
T ×J×Ce
. Finally, we apply a linear transformation to E to estimate 3D motion Xˆ ∈ R
T ×J×Cout . Here,
T denotes the sequence length, and J denotes the number
of body joints. Cin, Cf
, Ce, and Cout denote the channel
numbers of input, feature, embedding, and output respectively. We first introduce the basic building blocks of DSTformer, i.e. Spatial and Temporal Blocks with Multi-Head
Self-Attention (MHSA), and then explain the DSTformer
architecture design.
Spatial Block. Spatial MHSA (S-MHSA) aims at modeling the relationship among the joints within the same time
step. It is defined as
S-MHSA(QS, KS, VS) = [head1; ...; headh]WP
S
,
headi = softmax(
Qi
S
(Ki
S
)
′
√
dK
)Vi
S
,
(1)
where WP
S
is a projection parameter matrix, h is the number
of the heads, i ∈ 1, . . . , h, and ′ denotes matrix transpose.
We utilize self-attention to get the query QS, key KS, and
value VS from input per-frame spatial feature FS ∈ R
J×Ce
for each headi
,
Q
i
S = FSW(Q,i)
S
, K
i
S = FSW(K,i)
S
, V
i
S = FSW(V,i)
S
, (2)
where W(Q,i)
S
, W(K,i)
S
, W(V,i)
S
are projection matrices, and
dK is the feature dimension of KS. We apply S-MHSA to
features of different time steps in parallel. Residual connection and layer normalization (LayerNorm) are used to the
S-MHSA result, which is further fed into a multilayer perceptron (MLP), and followed by a residual connection and
LayerNorm following [112]. We denote the entire spatial
block with MHSA, LayerNorm, MLP, and residual connections by S.
Temporal Block. Temporal MHSA (T-MHSA) aims at
modeling the relationship across the time steps for a body
joint. Its computation process is similar with S-MHSA except that the MHSA is applied to the per-joint temporal
feature FT ∈ R
T ×Ce and parallelized over the spatial dimension.
T-MHSA(QT, KT, VT) = [head1; ...; headh]WP
T
,
headi = softmax(
Qi
T
(Ki
T
)
′
√
dK
)Vi
T
,
(3)
where i ∈ 1, . . . , h, QT, KT, VT are computed similar with
Formula 2. We denote the entire temporal block by T .