#include "inmemdb/store.h"
#include <new>

//FNV-1a hash function fast and good dist
uint64_t Store::hash_function(const char* key)const{
    uint64_t hash=14695981039346656037ULL;
    while(*key){
        hash^=static_cast<uint64_t>(*key++);//
        hash*=1099511628211ULL;
    }
    return hash;
}
Store::Store(size_t initial_capacity):capacity_(initial_capacity),size_(0){
    table_=static_cast<Entry*>(aligned_alloc(64,capacity_*sizeof(Entry)));
    if(!table_)
    {
        throw std::bad_alloc();
    }
    for(size_t i=0;i<capacity_;++i)
    {
        new (&table_[i]) Entry();
    }
}
Store::~Store()
{
    if(table_)
    {
        free(table_);
    }
}

size_t Store::find_slot(const char* key,uint64_t hash)const{
    size_t index=hash % capacity_;
    size_t probe=0;
    while(table_[index].occupied) //look for an empty slot or matching key
    {
        if(table_[index].hash==hash&&strcmp(table_[index].key,key)==0)
        {
            return index;//found
        }
        probe++;
        index=(hash+probe)%capacity_;//linear probing
        if(probe>=capacity_)//full table it shouldnt never verify with proper resizing
        {
            return capacity_; //indicate not found
        }
    }
    return index; //empty slot found
}

void Store::put(const char* key,const char* value){
    if(static_cast<double>(size_)/capacity_>0.7)
    {
        resize();
    }
    uint64_t hash=hash_function(key);
    size_t index= find_slot(key,hash);
    if(index>=capacity_)
    {
        return;
    }
    if(!table_[index].occupied)
    {
        size++;
    }
    //copying data
    strncpy(table_[index].key,key,63); //copy key with max length 63 
    table_[index].key[63]='\0';  
    strncpy(table_[index].value,value,255);
    table_[index].value[255]='\0';
    table_[index].hash=hash;
    table_[index].occupied=true;
}

const char* Store::get(const char* key)const{
    uint64_t hash=hash_function(key);
    size_t index=find_slot(key,hash);
    if(index<capacity_&&table_[index].occupied)
    {
        return table_[index].value;
    }
    return nullptr; //not found
}
bool Store::contains(const char* key)const{
    uint64_t hash=hash_function(key);
    size_t index=find_slot(key,hash);
    return index<capacity_&&table_[index].occupied;
}

void Store::remove(const char*key)
{
    uint64_t hash=hash_function(key);
    size_t index=find_slot(key,hash);
    if(index<capacity_&&table_[index].occupied)
    {
        table_[index].occupied=false;
        size_--;
        //a better approach is needed we need to rehash the following entries to fix 
        //probing chains THIS IS IMPORTANT BUT OMITED FOR NOW
    }
}

void Store:resize()
{
    size_t old_capacity=capacity_;
    Entry* old_table=table_;

    capacity_*=2;
    table_=static_cast<Entry*>(aligned_alloc(64,capacity_*sizeof(Entry)));
    if(!table_)
    {
        table=old_capacity;
        capacity_=old_capacity;
        return;
    }
    for(size_t i=0;i<capacity_;++i)
    {
        new (&table_[i]) Entry();
    }
    //Rehash all entries
    size_=0;
    for(size_t i =0;i<old_capacity;++i)
    {
        if(old_table[i].occupied)
        {
            put(old_table[i].key,old_table[i].value);
        }
    }
    free(old_table);
}