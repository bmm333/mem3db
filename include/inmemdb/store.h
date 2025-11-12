#pragma once
#include <cstring>
#include <cstdint>
#include <cstdlib>


class Store{
public:
    Store(size_t initial_capacity=1024);
    ~Store();
    
    Store(const Store&)=delete;
    Store& operator=(cosnt Store&)=delete;

    void put(const char* key,const char* value);
    const char* get(const char* key)const;
    bool contains(const char* key)const;
    void remove(const char* key)const;
    
    size_t size()const {return size_;}
    size_t capacity()const{return capacity_;}
private:
    struct Entry{
        char key[64];
        char value[256];
        uint64_t hash;
        bool occupied;

        Entry():hash(0),occupied(false){
            key[0]='\0';
            value[0]='\0';
        }
    }__attribute__((aligned(64)));
    Entry* table_;
    size_t capacity_;
    size_t size_;

    uint64_t hash_function(const char* key)const;
    size_t find_slot(const char* key,uint64_t hash)const;
    void resize();
}