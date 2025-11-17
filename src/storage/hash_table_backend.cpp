#include "inmemdb/storage/hash_table_backend.h"
#include <cstdlib>
#include <cstring>
#include <new>

namespace inmemdb{
    //Entry constructor
    HashTableBackend::Entry::Entry():hash(0),key_len(0),value_len(0),flags(0)
    {
        inline_key[0]='\0';
        inline_value[0]='\0';
    } // TODO: Canonical zeroing in constructor!!

    //Entry destructor
    HashTableBackend::Entry::~Entry(){
        if(key_is_external()&&external_key)
        {
            free(external_key);
            external_key=nullptr;
        }
        if(value_is_external()&&external_value)
        {
            free(external_value);
            external_value=nullptr;
        }
    }
    //HashTableBackend constructor
    HashTableBackend::HashTableBackend(const HashTableConfig& config)
    : config_(config),capacity_(config.initial_capacity),size_(0),inline_count_(0){
        table_=static_cast<Entry*>(alligned_alloc(64,capacity_*sizeof(Entry)));
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
        }
    }
}
