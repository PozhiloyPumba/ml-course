
def flatten(l):
    return [item for sublist in l for item in sublist]

def remove_tech_tokens(mystr, tokens_to_remove=['<eos>', '<sos>', '<unk>', '<pad>']):
    return [x for x in mystr if x not in tokens_to_remove]


def get_text(x, TRG_vocab):
    text = [TRG_vocab.itos[token] for token in x]
    try:
        end_idx = text.index('<eos>')
        text = text[:end_idx]
    except ValueError:
        pass
    text = remove_tech_tokens(text)
    if len(text) < 1:
        text = []
    return text


def generate_translation(src, trg, model, TRG_vocab):
    model.eval()

    output = model(src, trg, 0) #turn off teacher forcing
    output = output.argmax(dim=-1).cpu().numpy()

    original = get_text(list(trg[:,0].cpu().numpy()), TRG_vocab)
    generated = get_text(list(output[1:, 0]), TRG_vocab)
    
    print('Original: {}'.format(' '.join(original)))
    print('Generated: {}'.format(' '.join(generated)))
    print()

def generate_translation_bpe(src, trg, model, TRG_vocab, model_trg):
    model.eval()

    output = model(src, trg, 0) #turn off teacher forcing
    output = output.argmax(dim=-1).cpu().numpy()

    original = get_text(list(trg[:,0].cpu().numpy()), TRG_vocab)
    generated = get_text(list(output[1:, 0]), TRG_vocab)

    print('Original: {}'.format(model_trg.decode(original)))
    print('Generated: {}'.format(model_trg.decode(generated)))
    print()

from nltk.translate.bleu_score import corpus_bleu
import tqdm
import torch

def validate(test_iterator, model, TRG, model_tokenizer, gen_tr, mute = False):
    batch = next(iter(test_iterator))
    if not mute:
        for idx in range(5):
            src = batch.src[:, idx:idx+1]
            trg = batch.trg[:, idx:idx+1]
            if gen_tr is generate_translation:
                gen_tr(src, trg, model, TRG.vocab)
            else:
                gen_tr(src, trg, model, TRG.vocab, model_tokenizer)

    original_text = []
    generated_text = []
    model.eval()
    with torch.no_grad():
        for i, batch in tqdm.tqdm(enumerate(test_iterator)):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0) #turn off teacher forcing

            #trg = [trg sent len, batch size]
            #output = [trg sent len, batch size, output dim]

            output = output.argmax(dim=-1)

            original_text.extend([get_text(x, TRG.vocab) for x in trg.cpu().numpy().T])
            generated_text.extend([get_text(x, TRG.vocab) for x in output[1:].detach().cpu().numpy().T])
    return corpus_bleu([[text] for text in original_text], generated_text) * 100


import torch.nn as nn
import torch.optim as optim

def initModel(model, PAD_IDX):
    def init_weights(m):
        # <YOUR CODE HERE>
        for name, param in m.named_parameters():
            nn.init.uniform_(param, -0.08, 0.08)

    model.apply(init_weights)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f'The model has {count_parameters(model):,} trainable parameters')

    optimizer = optim.Adam(model.parameters())
    criterion = nn.CrossEntropyLoss(ignore_index = PAD_IDX)
    return model, optimizer, criterion