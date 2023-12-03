//
// Created by 咸的鱼 on 2023/12/3.
//

#ifndef MLLM_TRIE_HPP
#define MLLM_TRIE_HPP
#include "vector"
#include "unordered_map"
#include "string"
class TrieIterator;
namespace mllm {
template <typename Value>
class TrieIterator;
template <typename Value>
class Trie {
    Node *root;
    public:
    Trie():root(new Node()) {}
void insert(const std::vector<Value> &key) {
        auto node = root;
        for (auto ch : key) {
            auto next = node->children.find(ch);
            if (next == node->children.end()) {
                auto new_node = new Node();
                node->children[ch] = new_node;
                node = new_node;
            } else {
                node = next->second;
            }
        }
        node->is_leaf = true;
    }
    TrieIterator<Value> iterator() {
        return TrieIterator<Value>(root, std::vector<char>());
    }
    TrieIterator<Value> commonPrefixSearch(const std::vector<Value> &labels) {
        auto node = root;
        return TrieIterator<Value>(node, labels);
    }

    struct Node {
        std::unordered_map<Value, Node *> children;
//        Value value;
        bool is_leaf = false;
    };
};
template <typename Value>
class TrieIterator {
public:
    typename Trie<Value>::Node *node;
    std::vector<char> path;
    typename std::vector<Value>::iterator iter;
    TrieIterator(typename Trie<Value>::Node *node, std::vector<Value> labels) : node(node), iter(labels.begin()) {
        path.clear();
    }
    std::vector<char> next() {
        while (iter != path.end()) {
            auto next = node->children.find(*iter);
            if (next != node->children.end()) {
                node = next->second;
                path.push_back(*iter);
                if (node->is_leaf) {
                    iter++;
                    return path;
                }
            }
            iter++;
        }
        return {};

    }



};
} // namespace mllm

#endif // MLLM_TRIE_HPP
