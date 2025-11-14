#pragma once
#include "storage_backend.h"
#include <cstdint>

namespace inmemdb{
    class HashTableBackend: public IStorageBackend{
        public:
            explicit HashTableBackend(size_t initial_capacity=16384);
            ~HashTableBackend()override;
            //disabling copy
            HashTableBackend(const HashTableBackend)=delete;
            HashTableBackend& operator=(const HashTableBackend&)=delete;
            //istoragebackend interface
            void put(const char* key,const char* values)override;
            const char* get(const char* key)const override;
            bool contains(const char* key)const override;
            void remove(const char* key)override;

            size_t size() const override{return size_;}
            size_t capacity() const override{return capacity_;}
            void foreach(void(*callback)(const char* key,const char* value,void* ctx),void* ctx)const override;
        private:
            struct Entry{
                char key[64];
                char value[256];
                uint64_t hash;
                bool occupied;
                Entry();
            }__attribute__((aligned(64))); //cache line alignment
            Entry* table_;
            size_t capacity_;
            size_t size_;

            uint64_t hash_function(const char* key)const;
            size_t find_slot(const char* key,uint64_t hash)const;
            void resize();
    };
}//namespace inmemdb