#pragma once
#include "storage_backend.h"
#include <cstdint>
#include <cstring>

namespace inmemdb{
    struct HashTableConfig{
        size_t initial_capacity = 16384;
        double max_load_factor = 0.75;
        size_t inline_key_threshold = 56;
        size_t inline_value_threshold = 56;
        enum ProbingStrategy{
            LINEAR,
            QUADRATIC,
            DOUBLE_HASH
        };
        ProbingStrategy probing = ProbingStrategy::LINEAR;
};

class HashTableBackend : public IStorageBackend{
public:
        explicit HashTableBackend(const HashTableConfig& config={});
        ~HashTableBackend()override;
        //no copy/move
        HashTableBackend(const HashTableBackend&)=delete;
        HashTableBackend& operator=(const HashTableBackend&)=delete;
        HashTableBackend(HashTableBackend&&)=delete;
        HashTableBackend& operator=(HashTableBackend&&)=delete;

        //IstorageBackend interface
        void put(const char* key,const char* value)override;
        const char* get(const char* key) const override;
        bool contains(const char* key)const override;
        void remove(const char* key) override;

        size_t size() const override{return size_;}
        size_t capacity() const override{return capacity_;}
        void foreach(void(*callback)(const char* key,const char* value,void* ctx),void* ctx) const override;
        //Statistics
        double load_factor()const {return static_cast<double>(size_)/capacity_;}
        size_t inline_count()const {return inline_count_;}
        size_t external_count()const {return size_-inline_count_;}
        
private:
        //hybrid storage  entry - 128bytes (using 2 cache lines)
        struct Entry{
            //First cache line 64 bytes
            union{
                char inline_key[56]; //small key inline stored
                char* external_key; //large key alocated externally
            };
            uint64_t hash; //8bytes

            //Second cache line (64bytes)
            union{
                char inline_value[56]; //small value inline stored
                char* external_value; //large value alocated externally
            };
            uint16_t key_len; //2 bytes
            uint16_t value_len; //2 bytes
            uint8_t flags; //1 byte bit : occupied bit 1: key_External bit 2: value_External
            uint8_t padding[3];

            Entry() noexcept;
            ~Entry() noexcept;

            //Flag helpers
            //Flags: bit 0=occupied, bit 1=key_external, bit 2=value_external , bit 3 =tombstone
            inline bool is_occupied()const noexcept{return flags & 0x1;}
            inline void set_occupied(bool v) noexcept{
                flags=v?(flags|0x1):(flags & ~0x1);
            }
            inline bool is_tombstone() const noexcept { return flags & 0x08; }
            inline void set_tombstone(bool v) noexcept {
                flags = v ? (flags | 0x08) : (flags & ~0x08);
            }
            inline bool key_is_external() const{return flags & 0x02;}
            inline void set_key_external(bool v)
            {
                flags=v?(flags|0x02):(flags & ~0x02);
            }
            inline bool value_is_external()const {return flags & 0x04;}
            inline void set_value_external(bool v)
            {
                flags=v?(flags|0x04):(flags & ~0x04);
            }
            
            //get actual key/value 
            inline const char* get_key()const{
                return key_is_external()?external_key:inline_key;
            }
            inline const char* get_value()const{
                return value_is_external()?external_value:inline_value;
            }
        } __attribute__((aligned(64)));
        
        static_assert(sizeof(Entry)==128,"Entry size must be 128 bytes");
        static_assert(alignof(Entry) == 64, "Entry must be 64-byte aligned"); //ensuring cache line alignment
        HashTableConfig config_;
        Entry* table_;
        size_t capacity_;
        size_t size_;
        size_t inline_count_; //track inline vs external for stats

        //Core HT OP
        uint64_t hash_function(const char* key)const noexcept;
        size_t probe_index(uint64_t hash,size_t attempt)const noexcept;
        size_t find_slot(const char* key,uint64_t hash)const noexcept;
        size_t find_slot_for_insert(uint64_t hash)const noexcept;
        void resize();

        //Entry managment
        void set_entry(Entry& entry,const char* key,const char* value);
        void clear_entry(Entry& entry) noexcept;
        
    };
}