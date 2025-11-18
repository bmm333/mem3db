#include "inmemdb/storage/hash_table_backend.h"
#include <cstdlib>
#include <cstring>
#include <new>
#include <stdexcept>

namespace inmemdb{
    //Entry constructor
    HashTableBackend::Entry::Entry() noexcept
    :hash(0),key_len(0),value_len(0),flags(0)
    {
        std::memset(inline_key,0,sizeof(inline_key));
        std::memset(inline_value,0,sizeof(inline_value));
    } //Done

    //Entry destructor
    HashTableBackend::Entry::~Entry() noexcept{
        if (key_is_external() && external_key) {
        free(external_key);
        }
        if (value_is_external() && external_value) {
            free(external_value);
        }
    }
    //HashTableBackend constructor
    HashTableBackend::HashTableBackend(const HashTableConfig& config)
    : config_(config)
    , capacity_(config.initial_capacity)
    , size_(0)
    , inline_count_(0){
        //Allocating cache-aligned table
        table_=static_cast<Entry*>(aligned_alloc
            (64,capacity_*sizeof(Entry))
        );
        if(!table_){
            throw std::bad_alloc();
        }
        //Initialize all entries
        for(size_t i=0;i<capacity_;i++)
        {
            new (&table_[i]) Entry();
        }
    }
    //HashTableBackend destructor
    HashTableBackend::~HashTableBackend(){
        if(table_)
        {
            //Destructors will free external allocations
            for(size_t i=0;i<capacity_;i++)
            {
                table_[i].~Entry();
            }
            free(table_);
            table_=nullptr;
        }
    }
    //FNV-1a hash function , surley this is going to be replaced later for gpu friendly.
    uint64_t HashTableBackend::hash_function(const char* key) const noexcept {
        uint64_t hash = 14695981039346656037ULL;
        while (*key) {
            hash ^= static_cast<uint64_t>(*key++);
            hash *= 1099511628211ULL;
        }
        return hash ? hash : 1;
    }
    size_t HashTableBackend::probe_index(uint64_t hash, size_t attempt) const noexcept {
        switch (config_.probing) {
            case HashTableConfig::ProbingStrategy::LINEAR:
                return (hash + attempt) % capacity_;
            
            case HashTableConfig::ProbingStrategy::QUADRATIC:
                return (hash + attempt * attempt) % capacity_;
            
            case HashTableConfig::ProbingStrategy::DOUBLE_HASH: { 
                uint64_t hash2 = 1 + (hash % (capacity_ - 1));
                return (hash + attempt * hash2) % capacity_;
            }
            
            default:
                return (hash + attempt) % capacity_;
        }
    }
    //Slot for lookup , Tombestone supported
    size_t HashTableBackend::find_slot(const char* key,uint64_t hash)const noexcept{
        size_t attempt=0;
        while (attempt < capacity_) {
            size_t index = probe_index(hash, attempt);
            const Entry& entry = table_[index];
            //Empty slot key not found
            if (!entry.is_occupied() && !entry.is_tombstone()) {
                return capacity_; // Not found
            }
            //Found matching key
            if (entry.is_occupied() && 
                entry.hash == hash && 
                std::strcmp(entry.get_key(), key) == 0) {
                return index;
            }
            // Go past tombstones
            ++attempt;
        }
        return capacity_;
    }
    //Find slot for insertion , Tombestone supported
    size_t HashTableBackend::find_slot_for_insert(uint64_t hash) const noexcept {
        size_t attempt = 0;
        size_t first_tombstone = capacity_;
        while (attempt < capacity_) {
            size_t index = probe_index(hash, attempt);
            const Entry& entry = table_[index];
            //empty slot
            if (!entry.is_occupied() && !entry.is_tombstone()) {
                //Use tombstone if we found one earlier
                return (first_tombstone < capacity_) ? first_tombstone : index;
            }
            // Remember first tombstone
            if (entry.is_tombstone() && first_tombstone == capacity_) {
                first_tombstone = index;
            } 
            ++attempt;
        }
        return first_tombstone; // Use tombstone or indicate full (which shouldn't happen since we resize before a certain load factor)
    }
    //Set tentry data
    void HashTableBackend::set_entry(Entry& entry, const char* key, const char* value) {
        const size_t key_len=std::strlen(key);
        const size_t value_len=std::strlen(value);

        //clear old data if
        if(entry.is_occupied()){
            entry.~Entry();
        }
        //Unions have to be zeroed before use this is critical
        std::memset(&entry.inline_key,0,sizeof(entry.inline_key));
        std::memset(&entry.inline_value,0,sizeof(entry.inline_value));
        //Store key (inline or external)
        if (key_len < config_.inline_key_threshold) {
            std::memcpy(entry.inline_key, key, key_len + 1);
            entry.set_key_external(false);
        } else {
            entry.external_key = static_cast<char*>(std::malloc(key_len + 1));
            if (!entry.external_key) {
                throw std::bad_alloc();
            }
            std::memcpy(entry.external_key, key, key_len + 1);
            entry.set_key_external(true);
        }
        if (value_len < config_.inline_value_threshold) {
            std::memcpy(entry.inline_value, value, value_len + 1);
            entry.set_value_external(false);
            ++inline_count_;
        } else {
            entry.external_value = static_cast<char*>(std::malloc(value_len + 1));
            if (!entry.external_value) {
                //Cleanup key on failure
                if (entry.key_is_external()) {
                    free(entry.external_key);
                }
                throw std::bad_alloc();
            }
            std::memcpy(entry.external_value, value, value_len + 1);
            entry.set_value_external(true);
        }
        entry.hash = hash_function(key);
        entry.key_len = static_cast<uint16_t>(key_len);
        entry.value_len = static_cast<uint16_t>(value_len);
        entry.set_occupied(true);
        entry.set_tombstone(false);
    }

    //Entry cleanup
    void HashTableBackend::clear_entry(Entry& entry) noexcept {
        if (entry.key_is_external() && entry.external_key) {
        free(entry.external_key);
        entry.external_key = nullptr;
        }
        if (entry.value_is_external() && entry.external_value) {
            free(entry.external_value);
            entry.external_value = nullptr;
        } else if (entry.is_occupied() && !entry.value_is_external()) {
            --inline_count_;
        }
        entry.set_occupied(false);
        entry.flags = 0;
    }

    //put op
    void HashTableBackend::put(const char* key, const char* value) {
        if (!key || !value) {
            throw std::invalid_argument("key and value cannot be null");
        }
        
        if (static_cast<double>(size_ + 1) / capacity_ > config_.max_load_factor) {
            resize();
        }
        
        uint64_t hash = hash_function(key);
        
        // Check if key already exists
        size_t index = find_slot(key, hash);
        if (index < capacity_) {
            // Update existing
            set_entry(table_[index], key, value);
            return;
        }
        
        // Find slot for new key
        index = find_slot_for_insert(hash);
        if (index >= capacity_) {
            throw std::runtime_error("hash table full");
        }
        
        set_entry(table_[index], key, value);
        ++size_;
    }
    //get op
    const char* HashTableBackend::get(const char* key) const {
        if (!key) return nullptr;
        
        uint64_t hash = hash_function(key);
        size_t index = find_slot(key, hash);
        
        if (index < capacity_ && table_[index].is_occupied()) {
            return table_[index].get_value();
        }
        
        return nullptr;
    }
    //contains op
    bool HashTableBackend::contains(const char* key) const {
            return get(key) != nullptr;
        }
        //remove op
        void HashTableBackend::remove(const char* key) {
        if (!key) return;
        
        uint64_t hash = hash_function(key);
        size_t index = find_slot(key, hash);
        
        if (index < capacity_ && table_[index].is_occupied()) {
            clear_entry(table_[index]);
            table_[index].set_tombstone(true);
            --size_;
        }
    }
    //foreach op
    void HashTableBackend::foreach(void(*callback)(const char* key, const char* value, void* ctx), 
                                    void* ctx) const {
        if (!callback) return;
        
        for (size_t i = 0; i < capacity_; ++i) {
            const Entry& entry = table_[i];
            if (entry.is_occupied() && !entry.is_tombstone()) {
                callback(entry.get_key(), entry.get_value(), ctx);
            }
        }
    }
    //resize op
    void HashTableBackend::resize() {
        size_t old_capacity = capacity_;
        Entry* old_table = table_;
        //double capacity
        capacity_ *= 2;
        
        //allocate new table
        table_ = static_cast<Entry*>(
            aligned_alloc(64, capacity_ * sizeof(Entry))
        );
        
        if (!table_) {
            //restore old state
            table_ = old_table;
            capacity_ = old_capacity;
            throw std::bad_alloc();
        }
        //init new table
        for (size_t i = 0; i < capacity_; ++i) {
            new (&table_[i]) Entry();
        }
        //rehash all entries
        size_t old_size = size_;
        size_ = 0;
        inline_count_ = 0;
        for (size_t i = 0; i < old_capacity; ++i) {
            const Entry& old_entry = old_table[i];
            if (old_entry.is_occupied() && !old_entry.is_tombstone()) {
                put(old_entry.get_key(), old_entry.get_value());
            }
        }
        //free old table
        for (size_t i = 0; i < old_capacity; ++i) {
            old_table[i].~Entry();
        }
        free(old_table);
    }
}
