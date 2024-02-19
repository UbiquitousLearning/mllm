//
// Created by Rongjie Yi on 24-2-19.
//

#ifndef MODELING_CLIP_HPP
#define MODELING_CLIP_HPP

#include "models/vit/modeling_vit.hpp"
#include "configuration_clip.hpp"

class ClipVisionEmbedding final: public Module, public ClipConfig {
  Convolution2D patch_embedding = Convolution2D(3, hidden_dim, {patch, patch}, {patch, patch}, VALID, true, patch_embedding_name);
  Parameter cls_token = Parameter(1, 1, 1, hidden_dim, cls_token_name);
  Parameter position_ids = Parameter(1, int(img_hw/patch) * int(img_hw/patch) + 1, 1, 1,position_ids_name);
  Embedding position_embedding = Embedding( int(img_hw/patch) * int(img_hw/patch) + 1, hidden_dim, position_embeddings_name);

  vector<Tensor> Forward(vector<Tensor> inputs) override {
    auto embd = patch_embedding(inputs[0]);
    embd = embd.transpose(SEQUENCE, DIMENSION);
    embd = embd.flatten(HEAD, SEQUENCE);
    embd = Tensor::cat({cls_token(), embd}, SEQUENCE);
    embd = position_embedding(position_ids()) + embd;
    return {embd};
  }
};

class CLipVisionModel final: public Module, public ClipConfig {
  ClipVisionEmbedding embedding = ClipVisionEmbedding();
  vector<ViTBlock> blocks = List<ViTBlock>(block_num);
  LayerNorm norm = LayerNorm(hidden_dim, true, 1e-6, post_norm_name);

  vector<Tensor> Forward(vector<Tensor> inputs) override {
    auto x = embedding(inputs)[0];
    for (auto &block : blocks) {
      x = block({x})[0];
    }
    x = x.clip( {}, {}, {0}, {});
    x = norm(x);
    return {x};
  }
};


#endif //MODELING_CLIP_HPP
