import math
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

from einops import rearrange, repeat


class TrafficSignDataset(Dataset):
    def __init__(self, data_file):
        with open(data_file, 'rb') as f: # mở file data dưới dạng nhị phân
            data = pickle.load(f)
        if isinstance(data, dict): # nếu data là 1 dict, lấy 'features' và 'labels'
            self.images = data['features'] # lấy ảnh từ 'features'
            self.labels = data['labels'] # lấy nhãn từ 'labels'
        else:
            self.images, self.labels = data # nếu data là tuple, lấy ảnh và nhãn từ tuple
        self.images = torch.FloatTensor(self.images) / 255.0 # chuẩn hóa ảnh về khoảng [0, 1]
        self.labels = torch.LongTensor(self.labels) # chuyển nhãn thành tensor kiểu Long
        if self.images.shape[-1] == 3:
            self.images = self.images.permute(0, 3, 1, 2) 
            # đổi thứ tự các chiều của ảnh từ (batch_size, height, width, channels) sang (batch_size, channels, height, width) 
            # để phù hợp với định dạng Pytorch yêu cầu cho ảnh (channels first)
    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3, embed_dim=192):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim,
                                    kernel_size=patch_size, stride=patch_size)
        """
        Sử dụng Conv2d để:
            Cắt ảnh thành các patch không chồng lặp
            Đồng thời, biến mỗi patch thành 1 vector độ dài embed_dim
        """
    def forward(self, x):
        x = self.projection(x)
        """
        Ảnh đầu vào x có shape [B, C, H, W] (batch size, channels, height, width).
        Sau khi Conv2d, x có shape: [B, embed_dim, H_patch, W_patch],
        trong đó H_patch = H // patch_size, W_patch = W // patch_size.
        """
        x = x.flatten(2)
        """
        Gộp hai chiều không gian H_patch và W_patch lại thành 1 chiều:
        Shape sẽ trở thành: [B, embed_dim, n_patches].
        """
        x = x.transpose(1, 2)
        """
        Hoán đổi vị trí chiều embed_dim và n_patches, để có dạng:
        [B, n_patches, embed_dim] — tức mỗi ảnh được biểu diễn thành chuỗi n_patches vector có kích thước embed_dim.
        """
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=192, n_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim # số chiều của embedding vector đầu vào
        self.n_heads = n_heads # số "đầu" attention song song
        self.head_dim = embed_dim // n_heads # Mỗi head xử lý 1 phần của vector embedding (chia đều)
        assert embed_dim % n_heads == 0 # Đảm bảo rằng số chiều có thể chia đều cho số head
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim) 
        # Linear layer này ánh xạ x thành 3 tensor: Q (Query), K (Key), V (Value).
        # qkv sẽ có shape [B, N, 3 * embed_dim].
        self.projection = nn.Linear(embed_dim, embed_dim) 
        # Sau khi kết hợp các head xong, ta dùng linear layer này để "gộp" lại thành vector đầu ra cuối cùng
        self.dropout = nn.Dropout(dropout) # dùng để tránh overfitting trong attention weights

    def forward(self, x):
        b, n, _ = x.shape # b: batch_size, n: số lượng patches
        qkv = self.qkv(x) # Tính toán Q, K, V từ đầu vào x
        qkv = qkv.reshape(b, n, 3, self.n_heads, self.head_dim) # Reshape để tách Q, K, V
        qkv = qkv.permute(2, 0, 3, 1, 4) # Hoán đổi thứ tự các chiều để có dạng [Q/K/V, B, n_heads, n_patches, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2] # Tách Q, K, V ra từ tensor qkv, mỗi tensor có shape [B, n_heads, n_patches, head_dim]
        attention = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim) # Tính toán attention scores
        """
        Atttention(Q,K) = Q @ K^T / sqrt(head_dim)
        """
        attention = F.softmax(attention, dim=-1) # Áp dụng softmax để chuẩn hóa attention scores
        attention = self.dropout(attention) # Áp dụng dropout để tránh overfitting
        out = (attention @ v).transpose(1, 2).reshape(b, n, self.embed_dim)
        return self.projection(out)


class MLP(nn.Module):
    def __init__(self, embed_dim=192, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, hidden_dim) # Lớp Linear đầu tiên: tăng số chiều từ 192 → 768
        self.fc2 = nn.Linear(hidden_dim, embed_dim) # Lớp Linear thứ hai: giảm số chiều trở lại 768 → 192.
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        x = F.gelu(self.fc1(x))
        """
        Đầu tiên, dữ liệu đi qua fc1, rồi áp dụng hàm kích hoạt GELU (Gaussian Error Linear Unit), 
        giúp mô hình hóa phi tuyến tốt hơn ReLU trong ViT.
        """
        x = self.dropout(x) # Áp dụng dropout để tránh overfitting
        x = self.fc2(x) # Sau đó, dữ liệu đi qua fc2 để giảm số chiều về embed_dim
        return self.dropout(x) # Giảm thiểu overfitting bằng cách dropout sau lớp thứ hai


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim=192, n_heads=8, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadAttention(embed_dim, n_heads, dropout)
        self.mlp = MLP(embed_dim, hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
    def forward(self, x):
        x = x + self.attention(self.norm1(x)) # Thêm kết quả attention vào đầu vào, sau khi chuẩn hóa
        x = x + self.mlp(self.norm2(x)) # Thêm kết quả MLP vào đầu vào, sau khi chuẩn hóa
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_channels=3,
                 embed_dim=192, n_heads=8, hidden_dim=768,
                 n_layers=12, n_classes=43, dropout=0.1):
        super().__init__()
        self.patch_embedding = PatchEmbedding(img_size, patch_size,
                                              in_channels, embed_dim)
        self.n_patches = self.patch_embedding.n_patches
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(
            torch.randn(1, self.n_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, n_heads, hidden_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        b = x.size(0) # Lấy kích thước batch từ đầu vào x
        x = self.patch_embedding(x) # Chuyển đổi ảnh thành các patch embedding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        """
        self.cls_token là tham số học được, có shape [1, 1, embed_dim].
        repeat(...): lặp lại token này b lần để tạo một tensor [B, 1, embed_dim].
        Mỗi ảnh trong batch có một token đặc biệt [CLS] dùng để đại diện toàn bộ ảnh.
        Lưu ý: cls_token sẽ được dùng sau cùng để phân loại, không phải patch nào cả.
        """
        x = torch.cat([cls_tokens, x], dim=1)
        """
        Ghép cls_token vào đầu chuỗi patch embedding.
        x giờ có shape [B, N+1, D], trong đó:
        N+1: là số patch cộng thêm 1 cls_token.
        """
        x = self.dropout(x + self.pos_embedding)
        """
        Thêm Positional Embedding: vì attention không biết thứ tự vị trí, ta cộng embedding vị trí thủ công.
        self.pos_embedding: shape [1, N+1, D], được broadcast để cộng với x.
        Sau đó áp dụng Dropout để tránh overfitting.
        """
        for block in self.transformer_blocks:
            x = block(x)
        """
        Dữ liệu x sẽ đi qua n_layers TransformerBlock liên tiếp.
        Mỗi block gồm:
        LayerNorm → Multi-Head Attention → Residual
        LayerNorm → MLP → Residual
        """
        x = self.norm(x)
        # Sau khi đi qua các block, toàn bộ đầu ra được LayerNorm một lần nữa (chuẩn hóa theo chiều cuối).
        return self.head(x[:, 0])
        """
        x[:, 0]: lấy ra token đầu tiên trong chuỗi — chính là cls_token đã được biến đổi thông qua attention và MLP.
        self.head: là một lớp Linear → ánh xạ từ embed_dim → n_classes (ở đây là 43 class).
        Trả về dự đoán cuối cùng của mô hình cho từng ảnh trong batch.
        """
