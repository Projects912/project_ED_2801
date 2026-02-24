import os
import torch
import torch.nn as nn
import pandas as pd
import soundfile as sf
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from transformers import DebertaTokenizer, DebertaModel
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import f1_score, recall_score
from tqdm import tqdm
from PIL import Image


BASE_DIR = r"D:\arun work\melt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 32
EPOCHS = 50
FREEZE_EPOCHS = 3
LR = 1e-4
LATENT_DIM = 256
FIXED_AUDIO_LEN = 80000
MODALITY_DROPOUT_P = 0.2


emotion_encoder = LabelEncoder()
emotion_encoder.fit([
    "anger", "disgust", "fear",
    "joy", "neutral", "sadness", "surprise"
])

sentiment_encoder = LabelEncoder()
sentiment_encoder.fit(["negative", "neutral", "positive"])
class MELDDataset(Dataset):
    def __init__(self, csv_path, split):
        self.df = pd.read_csv(csv_path)
        self.split = split

        self.tokenizer = DebertaTokenizer.from_pretrained("microsoft/deberta-base")
        self.audio_processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")

        self.img_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        name = f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}"

        text = self.tokenizer(
            row.Utterance,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )

        audio_path = os.path.join(BASE_DIR, "audio", self.split, name + ".wav")
        audio_mask = 1
        try:
            wav, _ = sf.read(audio_path)
            if wav.ndim > 1:
                wav = wav.mean(axis=1)
        except:
            wav = np.zeros(FIXED_AUDIO_LEN)
            audio_mask = 0

        wav = torch.tensor(wav, dtype=torch.float32)
        wav = wav[:FIXED_AUDIO_LEN] if wav.size(0) > FIXED_AUDIO_LEN else \
              torch.nn.functional.pad(wav, (0, FIXED_AUDIO_LEN - wav.size(0)))

        audio = self.audio_processor(wav, sampling_rate=16000, return_tensors="pt")

        img_path = os.path.join(BASE_DIR, "frames", self.split, name + ".jpg")
        visual_mask = 1
        try:
            img = Image.open(img_path).convert("RGB")
            img = self.img_tf(img)
        except:
            img = torch.zeros(3, 224, 224)
            visual_mask = 0

        modality_mask = torch.tensor([1, audio_mask, visual_mask], dtype=torch.float32)

        emo = torch.tensor(emotion_encoder.transform([row.Emotion])[0], dtype=torch.long)
        sent = torch.tensor(sentiment_encoder.transform([row.Sentiment])[0], dtype=torch.long)

        return text, audio, img, modality_mask, emo, sent

class FeatureExtractors(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_enc = DebertaModel.from_pretrained("microsoft/deberta-base")
        self.audio_enc = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
        self.visual_enc = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.visual_enc.fc = nn.Identity()

    def forward(self, text, audio, image):
        t = self.text_enc(**text).last_hidden_state[:, 0]
        a = self.audio_enc(**audio).last_hidden_state.mean(dim=1)
        v = self.visual_enc(image)
        return t, a, v

class MISLS(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.t = nn.Linear(768, d)
        self.a = nn.Linear(768, d)
        self.v = nn.Linear(2048, d)

    def forward(self, t, a, v):
        return self.t(t), self.a(a), self.v(v)

class MaskedAttention(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.qkv = nn.Linear(d, d * 3)
        self.scale = d ** 0.5

    def forward(self, x):
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        w = torch.softmax((q @ k.transpose(-1, -2)) / self.scale, dim=-1)
        return w @ v, w

class BiGRU(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.gru = nn.GRU(d, d, bidirectional=True, batch_first=True)

    def forward(self, x):
        out, _ = self.gru(x)
        return out[:, -1]

class MultiTaskHead(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.emo = nn.Linear(d * 2, 7)
        self.sent = nn.Linear(d * 2, 3)

    def forward(self, h):
        return self.emo(h), self.sent(h)

class FullModel(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.ext = FeatureExtractors()
        self.misls = MISLS(d)
        self.attn = MaskedAttention(d)
        self.gru = BiGRU(d)
        self.head = MultiTaskHead(d)

    def forward(self, text, audio, img, mask):
        t, a, v = self.ext(text, audio, img)
        zt, za, zv = self.misls(t, a, v)

        if self.training:
            drop = (torch.rand(mask.size(0), 1, device=mask.device) > MODALITY_DROPOUT_P).float()
            mask = mask * torch.cat([torch.ones_like(drop), drop, drop], dim=1)

        zt, za, zv = zt * mask[:, 0:1], za * mask[:, 1:2], zv * mask[:, 2:3]
        fused = torch.stack([zt, za, zv], dim=1)
        out, attn = self.attn(fused)
        h = self.gru(out)
        emo_out, sent_out = self.head(h)
        return emo_out, sent_out

def train_epoch(model, loader, optimizer, crit_emo, crit_sent):
    model.train()
    losses, ye, pe, ys, ps = [], [], [], [], []

    for text, audio, img, mask, emo, sent in tqdm(loader):
        text = {k: v.squeeze(1).to(DEVICE) for k, v in text.items()}
        audio = {k: v.squeeze(1).to(DEVICE) for k, v in audio.items()}
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        emo, sent = emo.to(DEVICE), sent.to(DEVICE)

        optimizer.zero_grad()
        emo_out, sent_out = model(text, audio, img, mask)
        loss = crit_emo(emo_out, emo) + crit_sent(sent_out, sent)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        ye.extend(emo.cpu().numpy())
        pe.extend(torch.argmax(emo_out, 1).cpu().numpy())
        ys.extend(sent.cpu().numpy())
        ps.extend(torch.argmax(sent_out, 1).cpu().numpy())

    return (
        np.mean(losses),
        f1_score(ye, pe, average="macro"),
        recall_score(ye, pe, average="macro"),
        f1_score(ys, ps, average="macro")
    )

@torch.no_grad()
def evaluate(model, loader, crit_emo, crit_sent):
    model.eval()
    losses, ye, pe, ys, ps = [], [], [], [], []

    for text, audio, img, mask, emo, sent in loader:
        text = {k: v.squeeze(1).to(DEVICE) for k, v in text.items()}
        audio = {k: v.squeeze(1).to(DEVICE) for k, v in audio.items()}
        img, mask = img.to(DEVICE), mask.to(DEVICE)
        emo, sent = emo.to(DEVICE), sent.to(DEVICE)

        emo_out, sent_out = model(text, audio, img, mask)
        loss = crit_emo(emo_out, emo) + crit_sent(sent_out, sent)

        losses.append(loss.item())
        ye.extend(emo.cpu().numpy())
        pe.extend(torch.argmax(emo_out, 1).cpu().numpy())
        ys.extend(sent.cpu().numpy())
        ps.extend(torch.argmax(sent_out, 1).cpu().numpy())

    return (
        np.mean(losses),
        f1_score(ye, pe, average="macro"),
        recall_score(ye, pe, average="macro"),
        f1_score(ys, ps, average="macro")
    )

if __name__ == "__main__":

    csv_dir = os.path.join(BASE_DIR, "csv")
    train_csv = os.path.join(csv_dir, "train_sent_emo.csv")
    val_csv = os.path.join(csv_dir, "val_sent_emo.csv")
    test_csv = os.path.join(csv_dir, "test_sent_emo.csv")

    train_ds = MELDDataset(train_csv, "train")
    val_ds   = MELDDataset(val_csv, "val")
    test_ds  = MELDDataset(test_csv, "test")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    model = FullModel(LATENT_DIM).to(DEVICE)

    for p in model.ext.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

    emo_counts = np.bincount(
        train_ds.df.Emotion.map(lambda x: emotion_encoder.transform([x])[0])
    )
    emo_weights = torch.tensor(1.0 / (emo_counts + 1e-6)).to(DEVICE)

    crit_emo = nn.CrossEntropyLoss(weight=emo_weights)
    crit_sent = nn.CrossEntropyLoss()

    best_val_f1 = 0.0

    for epoch in range(EPOCHS):

        if epoch == FREEZE_EPOCHS:
            for p in model.ext.parameters():
                p.requires_grad = True

        tr = train_epoch(model, train_loader, optimizer, crit_emo, crit_sent)
        va = evaluate(model, val_loader, crit_emo, crit_sent)

        print(
            f"Epoch {epoch+1}/{EPOCHS} | "
            f"Train Emo F1: {tr[1]:.3f} | "
            f"Val Emo F1: {va[1]:.3f} | "
            f"Val Emo UAR: {va[2]:.3f}"
        )

        if va[1] > best_val_f1:
            best_val_f1 = va[1]
            torch.save(model.state_dict(), "best_model.pth")

    model.load_state_dict(torch.load("best_model.pth"))
    te = evaluate(model, test_loader, crit_emo, crit_sent)

    print(
        f"TEST RESULTS | "
        f"Emotion F1: {te[1]:.3f} | "
        f"Emotion UAR: {te[2]:.3f} | "
        f"Sentiment F1: {te[3]:.3f}"
    )


    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score, precision_score, confusion_matrix, matthews_corrcoef


    def evaluate_modality(model, loader, active):
        model.eval()
        y_true, y_pred = [], []

        for text, audio, img, mask, emo, sent in loader:
            text = {k: v.squeeze(1).to(DEVICE) for k, v in text.items()}
            audio = {k: v.squeeze(1).to(DEVICE) for k, v in audio.items()}
            img = img.to(DEVICE)
            emo = emo.to(DEVICE)

            mask_mod = torch.zeros_like(mask)
            for idx in active:
                mask_mod[:, idx] = 1
            mask_mod = mask_mod.to(DEVICE)

            emo_out, _ = model(text, audio, img, mask_mod)
            preds = torch.argmax(emo_out, 1)

            y_true.extend(emo.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, average="macro")
        rec = recall_score(y_true, y_pred, average="macro")
        f1 = f1_score(y_true, y_pred, average="macro")
        mcc = matthews_corrcoef(y_true, y_pred)

        cm = confusion_matrix(y_true, y_pred)
        tn = cm.sum() - (cm.sum(axis=1) + cm.sum(axis=0) - np.diag(cm))
        fp = cm.sum(axis=0) - np.diag(cm)
        spec = np.mean(tn / (tn + fp + 1e-6))

        return acc, prec, rec, f1, spec, mcc

    modalities = {
        "Text":[0],
        "Audio":[1],
        "Visual":[2],
        "Text+Audio":[0,1],
        "Text+Visual":[0,2],
        "Audio+Visual":[1,2],
        "Multi-modal":[0,1,2]
    }

    results = {m: evaluate_modality(model, test_loader, idx) for m, idx in modalities.items()}

    metrics_names = ["Accuracy","Precision","Recall","F1","Specificity","MCC"]
    fig, axes = plt.subplots(2,3, figsize=(14,8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics_names):
        vals = [results[m][i] for m in modalities]
        axes[i].barh(list(modalities.keys()), vals)
        axes[i].set_title(metric)
        axes[i].set_xlim(0,1)

    plt.tight_layout()
    plt.show()


    weights_sum = torch.zeros(3).to(DEVICE)
    count = 0

    for text, audio, img, mask, emo, sent in test_loader:
        text = {k: v.squeeze(1).to(DEVICE) for k, v in text.items()}
        audio = {k: v.squeeze(1).to(DEVICE) for k, v in audio.items()}
        img = img.to(DEVICE)

        t,a,v = model.ext(text,audio,img)
        zt,za,zv = model.misls(t,a,v)
        fused = torch.stack([zt,za,zv],dim=1)
        _,attn = model.attn(fused)
        weights_sum += attn.mean(dim=0).mean(dim=0)
        count += 1

    weights = (weights_sum/count).cpu().numpy()

    labels = ["Text","Audio","Visual"]
    angles = np.linspace(0,2*np.pi,len(labels),endpoint=False)
    weights = np.concatenate((weights,[weights[0]]))
    angles = np.concatenate((angles,[angles[0]]))

    plt.figure(figsize=(6,6))
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, weights)
    ax.fill(angles, weights, alpha=0.25)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    plt.title("Modality Attention Weights")
    plt.show()


    emotion_examples = {}
    for _, row in test_ds.df.iterrows():
        if row.Emotion not in emotion_examples:
            emotion_examples[row.Emotion] = row
        if len(emotion_examples) == 7:
            break

    plt.figure(figsize=(12,8))

    for i,(emo,row) in enumerate(emotion_examples.items()):
        name = f"dia{row.Dialogue_ID}_utt{row.Utterance_ID}"
        audio_path = os.path.join(BASE_DIR,"audio","test",name+".wav")
        wav,sr = sf.read(audio_path)
        if wav.ndim>1:
            wav = wav.mean(axis=1)
        t = np.linspace(0,len(wav)/sr,len(wav))

        plt.subplot(4,2,i+1)
        plt.plot(t,wav)
        plt.title(f"Emotion: {emo}")
        plt.xlabel("Time (s)")
        plt.ylabel("Amplitude")

    plt.tight_layout()
    plt.show()