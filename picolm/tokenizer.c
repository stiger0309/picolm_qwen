#include "tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ---- GGUF string reader (reused from model.c logic) ---- */

static uint64_t read_u64_at(const uint8_t **p) {
    uint64_t v;
    memcpy(&v, *p, 8);
    *p += 8;
    return v;
}

/* ---- Sorted index for binary search ---- */

static char **g_vocab_for_sort; /* global for qsort comparison */

static int cmp_sorted(const void *a, const void *b) {
    int ia = *(const int *)a;
    int ib = *(const int *)b;
    return strcmp(g_vocab_for_sort[ia], g_vocab_for_sort[ib]);
}

static int vocab_lookup(const tokenizer_t *t, const char *str, int len) {
    /* Binary search in sorted vocabulary */
    int lo = 0, hi = t->vocab_size - 1;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        int idx = t->sorted_idx[mid];
        int cmp = strncmp(t->vocab[idx], str, (size_t)len);
        if (cmp == 0) {
            /* Check exact length match */
            if (t->vocab[idx][len] == '\0') return idx;
            if (t->vocab[idx][len] > '\0') { hi = mid - 1; }
            else { lo = mid + 1; }
        } else if (cmp < 0) {
            lo = mid + 1;
        } else {
            hi = mid - 1;
        }
    }
    return -1; /* not found */
}

/* ---- GPT-2/Qwen byte<->unicode map used by byte-level BPE ---- */

static int g_qwen_u2b[1024];
static int g_qwen_b2u[256];
static int g_qwen_u2b_inited = 0;

static void init_qwen_u2b_map(void) {
    if (g_qwen_u2b_inited) return;

    for (int i = 0; i < (int)(sizeof(g_qwen_u2b) / sizeof(g_qwen_u2b[0])); i++) {
        g_qwen_u2b[i] = -1;
    }

    int extra = 0;
    for (int b = 0; b < 256; b++) {
        int keep = ((b >= 33 && b <= 126) || (b >= 161 && b <= 172) || (b >= 174 && b <= 255));
        int cp = keep ? b : (256 + extra++);
        g_qwen_b2u[b] = cp;
        if (cp >= 0 && cp < (int)(sizeof(g_qwen_u2b) / sizeof(g_qwen_u2b[0]))) {
            g_qwen_u2b[cp] = b;
        }
    }

    g_qwen_u2b_inited = 1;
}

static int utf8_encode_cp(uint32_t cp, char out[4]) {
    if (cp <= 0x7F) {
        out[0] = (char)cp;
        return 1;
    }
    if (cp <= 0x7FF) {
        out[0] = (char)(0xC0 | ((cp >> 6) & 0x1F));
        out[1] = (char)(0x80 | (cp & 0x3F));
        return 2;
    }
    out[0] = (char)(0xE0 | ((cp >> 12) & 0x0F));
    out[1] = (char)(0x80 | ((cp >> 6) & 0x3F));
    out[2] = (char)(0x80 | (cp & 0x3F));
    return 3;
}

static int utf8_next_cp(const char *s, int len, int *i, uint32_t *cp_out) {
    if (*i >= len) return 0;

    unsigned char c0 = (unsigned char)s[*i];
    if (c0 < 0x80) {
        *cp_out = (uint32_t)c0;
        (*i)++;
        return 1;
    }

    if ((c0 & 0xE0) == 0xC0 && *i + 1 < len) {
        unsigned char c1 = (unsigned char)s[*i + 1];
        if ((c1 & 0xC0) == 0x80) {
            *cp_out = ((uint32_t)(c0 & 0x1F) << 6) | (uint32_t)(c1 & 0x3F);
            *i += 2;
            return 1;
        }
    } else if ((c0 & 0xF0) == 0xE0 && *i + 2 < len) {
        unsigned char c1 = (unsigned char)s[*i + 1];
        unsigned char c2 = (unsigned char)s[*i + 2];
        if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80) {
            *cp_out = ((uint32_t)(c0 & 0x0F) << 12) |
                      ((uint32_t)(c1 & 0x3F) << 6) |
                      (uint32_t)(c2 & 0x3F);
            *i += 3;
            return 1;
        }
    } else if ((c0 & 0xF8) == 0xF0 && *i + 3 < len) {
        unsigned char c1 = (unsigned char)s[*i + 1];
        unsigned char c2 = (unsigned char)s[*i + 2];
        unsigned char c3 = (unsigned char)s[*i + 3];
        if ((c1 & 0xC0) == 0x80 && (c2 & 0xC0) == 0x80 && (c3 & 0xC0) == 0x80) {
            *cp_out = ((uint32_t)(c0 & 0x07) << 18) |
                      ((uint32_t)(c1 & 0x3F) << 12) |
                      ((uint32_t)(c2 & 0x3F) << 6) |
                      (uint32_t)(c3 & 0x3F);
            *i += 4;
            return 1;
        }
    }

    /* Invalid UTF-8 sequence: pass byte through. */
    *cp_out = (uint32_t)c0;
    (*i)++;
    return 1;
}

/* ---- Public API ---- */

int tokenizer_load(tokenizer_t *t, const model_t *m) {
    memset(t, 0, sizeof(*t));
    int vs = m->config.vocab_size;
    t->vocab_size = vs;
    t->bos_id = m->tok_bos_id;
    t->eos_id = m->tok_eos_id;
    t->mem_vocab_ptrs = (size_t)vs * sizeof(char *);
    t->mem_scores = (size_t)vs * sizeof(float);
    t->mem_sorted_idx = (size_t)vs * sizeof(int);

    /* Allocate vocab and scores arrays */
    t->vocab = (char **)calloc((size_t)vs, sizeof(char *));
    t->scores = (float *)calloc((size_t)vs, sizeof(float));
    t->sorted_idx = (int *)malloc((size_t)vs * sizeof(int));
    if (!t->vocab || !t->scores || !t->sorted_idx) {
        fprintf(stderr, "OOM allocating tokenizer\n");
        return -1;
    }

    /* Read vocab strings from GGUF metadata array */
    if (m->tok_tokens_data && m->tok_n_tokens > 0) {
        const uint8_t *p = (const uint8_t *)m->tok_tokens_data;
        uint64_t n = m->tok_n_tokens;
        if ((int)n > vs) n = (uint64_t)vs;

        for (uint64_t i = 0; i < n; i++) {
            uint64_t slen = read_u64_at(&p);
            /* Allocate and copy the string with null terminator */
            t->vocab[i] = (char *)malloc((size_t)slen + 1);
            if (t->vocab[i]) {
                memcpy(t->vocab[i], p, (size_t)slen);
                t->vocab[i][slen] = '\0';
                t->mem_vocab_strings += (size_t)slen + 1;
            }
            p += slen;
        }
    }

    /* Fill any remaining entries with empty strings */
    for (int i = 0; i < vs; i++) {
        if (!t->vocab[i]) {
            t->vocab[i] = (char *)calloc(1, 1);
            if (t->vocab[i]) {
                t->mem_vocab_strings += 1;
            }
        }
    }

    /* Read scores */
    if (m->tok_scores_data && m->tok_n_scores > 0) {
        uint64_t n = m->tok_n_scores;
        if ((int)n > vs) n = (uint64_t)vs;
        memcpy(t->scores, m->tok_scores_data, (size_t)n * sizeof(float));
    }

    /* Build sorted index */
    for (int i = 0; i < vs; i++) {
        t->sorted_idx[i] = i;
    }
    g_vocab_for_sort = t->vocab;
    qsort(t->sorted_idx, (size_t)vs, sizeof(int), cmp_sorted);
    t->mem_total = t->mem_vocab_ptrs + t->mem_scores + t->mem_sorted_idx + t->mem_vocab_strings;

    fprintf(stderr, "Tokenizer loaded: %d tokens, bos=%u, eos=%u\n",
            vs, t->bos_id, t->eos_id);
    return 0;
}

/*int tokenizer_encode(encode_ptr func, const tokenizer_t *t, const char *text, int *tokens, int max_tokens, int add_bos) {
    if (!func) return 0;
    return func(t, text, tokens, max_tokens, add_bos);
}

const char* tokenizer_decode(decode_ptr func, const tokenizer_t *t, int prev_token, int token) {
    if (!func) return NULL;
    return func(t, prev_token, token);
}*/

int tokenizer_encode_llama(const tokenizer_t *t, const char *text, int *tokens, int max_tokens, int add_bos) {
    int n_tokens = 0;

    if (add_bos && n_tokens < max_tokens) {
        tokens[n_tokens++] = (int)t->bos_id;
    }

    if (!text || !*text) return n_tokens;

    /* SentencePiece convention: replace spaces with ▁ (U+2581, UTF-8: E2 96 81)
     * and prepend ▁ at the start of the text.
     * Build a normalized copy: " Once upon a time" → "▁Once▁upon▁a▁time" */
    int text_len = (int)strlen(text);
    /* Worst case: every byte is a space → 3x expansion, plus leading ▁ */
    int norm_cap = text_len * 3 + 4;
    char *norm = (char *)malloc((size_t)norm_cap);
    int norm_len = 0;

    /* Add leading ▁ */
    norm[norm_len++] = (char)0xE2;
    norm[norm_len++] = (char)0x96;
    norm[norm_len++] = (char)0x81;

    for (int i = 0; i < text_len; i++) {
        if (text[i] == ' ') {
            norm[norm_len++] = (char)0xE2;
            norm[norm_len++] = (char)0x96;
            norm[norm_len++] = (char)0x81;
        } else {
            norm[norm_len++] = text[i];
        }
    }
    norm[norm_len] = '\0';

    /* Step 1: Convert normalized text to individual character tokens.
     * Each UTF-8 character (including ▁) gets looked up in the vocab. */
    /* Worst case: one token per byte of normalized text */
    int *merge_buf = (int *)malloc((size_t)(norm_len + 1) * sizeof(int));
    int merge_len = 0;

    for (int i = 0; i < norm_len; ) {
        /* Determine UTF-8 character length */
        int clen = 1;
        unsigned char c = (unsigned char)norm[i];
        if (c >= 0xF0) clen = 4;
        else if (c >= 0xE0) clen = 3;
        else if (c >= 0xC0) clen = 2;

        if (i + clen > norm_len) clen = norm_len - i;

        /* Try to find this character in vocab */
        int tok = vocab_lookup(t, norm + i, clen);
        if (tok >= 0) {
            merge_buf[merge_len++] = tok;
            i += clen;
        } else {
            /* Fall back to byte tokens: <0xHH> */
            char byte_tok[8];
            snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)norm[i]);
            tok = vocab_lookup(t, byte_tok, (int)strlen(byte_tok));
            if (tok >= 0) {
                merge_buf[merge_len++] = tok;
            }
            i++;
        }
    }
    free(norm);

    /* Step 2: BPE merge loop — iteratively find best adjacent pair */
    while (merge_len >= 2) {
        float best_score = -1e30f;
        int best_idx = -1;
        int best_tok = -1;

        for (int i = 0; i < merge_len - 1; i++) {
            /* Build the merged string */
            const char *s1 = t->vocab[merge_buf[i]];
            const char *s2 = t->vocab[merge_buf[i + 1]];
            int l1 = (int)strlen(s1);
            int l2 = (int)strlen(s2);

            /* Build concatenation in stack buffer */
            char merged[256];
            if (l1 + l2 >= (int)sizeof(merged)) continue;
            memcpy(merged, s1, (size_t)l1);
            memcpy(merged + l1, s2, (size_t)l2);
            merged[l1 + l2] = '\0';

            int tok = vocab_lookup(t, merged, l1 + l2);
            if (tok >= 0 && t->scores[tok] > best_score) {
                best_score = t->scores[tok];
                best_idx = i;
                best_tok = tok;
            }
        }

        if (best_idx < 0) break; /* no more merges possible */

        /* Apply the merge */
        merge_buf[best_idx] = best_tok;
        /* Shift left */
        for (int i = best_idx + 1; i < merge_len - 1; i++) {
            merge_buf[i] = merge_buf[i + 1];
        }
        merge_len--;
    }

    /* Copy to output */
    for (int i = 0; i < merge_len && n_tokens < max_tokens; i++) {
        tokens[n_tokens++] = merge_buf[i];
    }

    free(merge_buf);
    return n_tokens;
}

/* 辅助函数：对不含特殊 Token 的纯文本片段进行 BPE 编码 */
static int encode_bpe_segment(const tokenizer_t *t, const char *text, int len, int *tokens, int max_tokens) {
    if (len <= 0) return 0;
    init_qwen_u2b_map();

    /* Step 1: byte-level bytes_to_unicode 映射后的字符切分 */
    int *merge_buf = (int *)malloc((size_t)(len + 1) * sizeof(int));
    int merge_len = 0;

    for (int i = 0; i < len; i++) {
        unsigned char b = (unsigned char)text[i];
        uint32_t cp = (uint32_t)g_qwen_b2u[b];
        char mapped[4];
        int mapped_len = utf8_encode_cp(cp, mapped);

        int tok = vocab_lookup(t, mapped, mapped_len);
        if (tok >= 0) {
            merge_buf[merge_len++] = tok;
        } else {
            /* Fallback：直接用 <0xHH> byte token */
            char byte_tok[8];
            snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", b);
            tok = vocab_lookup(t, byte_tok, (int)strlen(byte_tok));
            if (tok >= 0) merge_buf[merge_len++] = tok;
        }
    }

    /* Step 2: BPE 合并循环 */
    while (merge_len >= 2) {
        float best_score = -1e30f;
        int best_idx = -1;
        int best_tok = -1;

        for (int i = 0; i < merge_len - 1; i++) {
            const char *s1 = t->vocab[merge_buf[i]];
            const char *s2 = t->vocab[merge_buf[i + 1]];
            int l1 = (int)strlen(s1);
            int l2 = (int)strlen(s2);

            char merged[512]; /* Qwen 的词条可能较长 */
            if (l1 + l2 >= (int)sizeof(merged)) continue;
            memcpy(merged, s1, (size_t)l1);
            memcpy(merged + l1, s2, (size_t)l2);
            merged[l1 + l2] = '\0';

            int tok = vocab_lookup(t, merged, l1 + l2);
            if (tok >= 0 && t->scores[tok] > best_score) {
                best_score = t->scores[tok];
                best_idx = i;
                best_tok = tok;
            }
        }

        if (best_idx < 0) break;
        merge_buf[best_idx] = best_tok;
        for (int i = best_idx + 1; i < merge_len - 1; i++) merge_buf[i] = merge_buf[i + 1];
        merge_len--;
    }

    int count = (merge_len < max_tokens) ? merge_len : max_tokens;
    for (int i = 0; i < count; i++) tokens[i] = merge_buf[i];
    
    free(merge_buf);
    return count;
}

int tokenizer_encode_qwen(const tokenizer_t *t, const char *text, int *tokens, int max_tokens, int add_bos) {
    int n_tokens = 0;

    if (add_bos && n_tokens < max_tokens) {
        tokens[n_tokens++] = (int)t->bos_id;
    }

    if (!text || !*text) return n_tokens;

    init_qwen_u2b_map();

    /* Convert raw UTF-8 bytes to GPT-2/Qwen bytes_to_unicode-mapped string. */
    int text_len = (int)strlen(text);
    int mangled_cap = text_len * 4 + 1;
    char *mangled = (char *)malloc((size_t)mangled_cap);
    if (!mangled) return n_tokens;

    int mlen = 0;
    for (int i = 0; i < text_len; i++) {
        unsigned char b = (unsigned char)text[i];
        uint32_t cp = (uint32_t)g_qwen_b2u[b];
        char tmp[4];
        int n = utf8_encode_cp(cp, tmp);
        if (mlen + n >= mangled_cap) break;
        memcpy(mangled + mlen, tmp, (size_t)n);
        mlen += n;
    }
    mangled[mlen] = '\0';

    /* Greedy longest-match over mapped string (matches cpp/tokenizer.hpp behavior). */
    int i = 0;
    while (i < mlen && n_tokens < max_tokens) {
        int best_len = 0;
        int best_tok = -1;

        int max_try = mlen - i;
        if (max_try > 48) max_try = 48;
        for (int try_len = max_try; try_len >= 1; --try_len) {
            int tok = vocab_lookup(t, mangled + i, try_len);
            if (tok >= 0) {
                best_len = try_len;
                best_tok = tok;
                break;
            }
        }

        if (best_tok >= 0) {
            tokens[n_tokens++] = best_tok;
            i += best_len;
        } else {
            /* Fallback: skip one UTF-8 codepoint in mapped stream. */
            uint32_t cp = 0;
            int old_i = i;
            utf8_next_cp(mangled, mlen, &i, &cp);
            if (i == old_i) i++;
        }
    }

    free(mangled);
    return n_tokens;
}

int tokenizer_encode_qwen_o(const tokenizer_t *t, const char *text, int *tokens, int max_tokens, int add_bos) {
    int n_tokens = 0;

    if (add_bos && n_tokens < max_tokens) {
        tokens[n_tokens++] = (int)t->bos_id;
    }

    if (!text || !*text) return n_tokens;

    /* SentencePiece convention: replace spaces with ▁ (U+2581, UTF-8: E2 96 81)
     * and prepend ▁ at the start of the text.
     * Build a normalized copy: " Once upon a time" → "▁Once▁upon▁a▁time" */
    int text_len = (int)strlen(text);
    /* Worst case: every byte is a space → 3x expansion, plus leading ▁ */
    int norm_cap = text_len * 3 + 4;
    char *norm = (char *)malloc((size_t)norm_cap);
    int norm_len = 0;

    

    for (int i = 0; i < text_len; i++) {
        if (text[i] == ' ') {
        } else {
            norm[norm_len++] = text[i];
        }
    }
    norm[norm_len] = '\0';

    /* Step 1: Convert normalized text to individual character tokens.
     * Each UTF-8 character (including ▁) gets looked up in the vocab. */
    /* Worst case: one token per byte of normalized text */
    int *merge_buf = (int *)malloc((size_t)(norm_len + 1) * sizeof(int));
    int merge_len = 0;

    for (int i = 0; i < norm_len; ) {
        /* Determine UTF-8 character length */
        int clen = 1;
        unsigned char c = (unsigned char)norm[i];
        if (c >= 0xF0) clen = 4;
        else if (c >= 0xE0) clen = 3;
        else if (c >= 0xC0) clen = 2;

        if (i + clen > norm_len) clen = norm_len - i;

        /* Try to find this character in vocab */
        int tok = vocab_lookup(t, norm + i, clen);
        if (tok >= 0) {
            merge_buf[merge_len++] = tok;
            i += clen;
        } else {
            /* Fall back to byte tokens: <0xHH> */
            char byte_tok[8];
            snprintf(byte_tok, sizeof(byte_tok), "<0x%02X>", (unsigned char)norm[i]);
            tok = vocab_lookup(t, byte_tok, (int)strlen(byte_tok));
            if (tok >= 0) {
                merge_buf[merge_len++] = tok;
            }
            i++;
        }
    }
    free(norm);

    /* Step 2: BPE merge loop — iteratively find best adjacent pair */
    while (merge_len >= 2) {
        float best_score = -1e30f;
        int best_idx = -1;
        int best_tok = -1;

        for (int i = 0; i < merge_len - 1; i++) {
            /* Build the merged string */
            const char *s1 = t->vocab[merge_buf[i]];
            const char *s2 = t->vocab[merge_buf[i + 1]];
            int l1 = (int)strlen(s1);
            int l2 = (int)strlen(s2);

            /* Build concatenation in stack buffer */
            char merged[256];
            if (l1 + l2 >= (int)sizeof(merged)) continue;
            memcpy(merged, s1, (size_t)l1);
            memcpy(merged + l1, s2, (size_t)l2);
            merged[l1 + l2] = '\0';

            int tok = vocab_lookup(t, merged, l1 + l2);
            if (tok >= 0 && t->scores[tok] > best_score) {
                best_score = t->scores[tok];
                best_idx = i;
                best_tok = tok;
            }
        }

        if (best_idx < 0) break; /* no more merges possible */

        /* Apply the merge */
        merge_buf[best_idx] = best_tok;
        /* Shift left */
        for (int i = best_idx + 1; i < merge_len - 1; i++) {
            merge_buf[i] = merge_buf[i + 1];
        }
        merge_len--;
    }

    /* Copy to output */
    for (int i = 0; i < merge_len && n_tokens < max_tokens; i++) {
        tokens[n_tokens++] = merge_buf[i];
    }

    free(merge_buf);
    return n_tokens;
}

const char *tokenizer_decode_qwen(const tokenizer_t *t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) return "";

    const char *str = t->vocab[token];
    static char clean_buf[4096];

    init_qwen_u2b_map();

    int slen = (int)strlen(str);
    int i = 0;
    int j = 0;

    while (i < slen && j < (int)sizeof(clean_buf) - 1) {
        if (str[i] == '<' && i + 5 < slen && str[i + 1] == '0' && str[i + 2] == 'x' && str[i + 5] == '>') {
            unsigned int val = 0;
            if (sscanf(str + i, "<0x%02X>", &val) == 1) {
                clean_buf[j++] = (char)val;
                i += 6;
                continue;
            }
        }

        int start = i;
        uint32_t cp = 0;
        utf8_next_cp(str, slen, &i, &cp);

        if (cp < (uint32_t)(sizeof(g_qwen_u2b) / sizeof(g_qwen_u2b[0])) && g_qwen_u2b[cp] >= 0) {
            clean_buf[j++] = (char)g_qwen_u2b[cp];
        } else {
            int n = i - start;
            if (n > (int)sizeof(clean_buf) - 1 - j) n = (int)sizeof(clean_buf) - 1 - j;
            memcpy(clean_buf + j, str + start, (size_t)n);
            j += n;
        }
    }

    clean_buf[j] = '\0';
    return clean_buf;
}

const char *tokenizer_decode_llama(const tokenizer_t *t, int prev_token, int token) {
    if (token < 0 || token >= t->vocab_size) return "";

    const char *str = t->vocab[token];

    /* Handle byte tokens: <0xHH> -> single byte */
    if (str[0] == '<' && str[1] == '0' && str[2] == 'x' && str[5] == '>') {
        /* Decode hex byte */
        static char byte_buf[2];
        unsigned int val = 0;
        for (int i = 3; i < 5; i++) {
            val <<= 4;
            char c = str[i];
            if (c >= '0' && c <= '9') val += (unsigned)(c - '0');
            else if (c >= 'A' && c <= 'F') val += (unsigned)(c - 'A' + 10);
            else if (c >= 'a' && c <= 'f') val += (unsigned)(c - 'a' + 10);
        }
        byte_buf[0] = (char)val;
        byte_buf[1] = '\0';
        return byte_buf;
    }

    /* Handle SentencePiece leading space marker "▁" -> " " */
    /* The "▁" character is U+2581, encoded as 0xE2 0x96 0x81 in UTF-8 */
    if ((unsigned char)str[0] == 0xE2 && (unsigned char)str[1] == 0x96 && (unsigned char)str[2] == 0x81) {
        /* Replace leading ▁ with space, but not after BOS */
        static char space_buf[256];
        if (prev_token == (int)t->bos_id) {
            /* After BOS, strip the leading space */
            int len = (int)strlen(str + 3);
            if (len >= (int)sizeof(space_buf)) len = (int)sizeof(space_buf) - 1;
            memcpy(space_buf, str + 3, (size_t)len);
            space_buf[len] = '\0';
            return space_buf;
        }
        space_buf[0] = ' ';
        int len = (int)strlen(str + 3);
        if (len >= (int)sizeof(space_buf) - 1) len = (int)sizeof(space_buf) - 2;
        memcpy(space_buf + 1, str + 3, (size_t)len);
        space_buf[1 + len] = '\0';
        return space_buf;
    }

    return str;
}

void tokenizer_free(tokenizer_t *t) {
    if (t->vocab) {
        for (int i = 0; i < t->vocab_size; i++) {
            free(t->vocab[i]);
        }
        free(t->vocab);
        t->vocab = NULL;
    }
    free(t->scores);
    t->scores = NULL;
    free(t->sorted_idx);
    t->sorted_idx = NULL;
    t->mem_vocab_ptrs = 0;
    t->mem_scores = 0;
    t->mem_sorted_idx = 0;
    t->mem_vocab_strings = 0;
    t->mem_total = 0;
}

size_t tokenizer_memory_bytes(const tokenizer_t *t) {
    return t ? t->mem_total : 0;
}
