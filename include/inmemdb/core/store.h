#pragma once
#include "inmemdb/storage/storage_backend.h"
#include "inmemdb/presistence/persistence_engine.h"
#include <shared_mutex>
#include <memory>

namespace inmemdb{
class Store{
public:
        Store(std::unique_ptr<IStorageBackend>backend,
        std::unique_ptr<IPersistenceEngine> presistence=nullptr);
        ~Store();

        //Def Public Api 
        void put(const char*key,const char* value);
        const char* get(const chr* key)const;
        bool contains(const char* key)const;
        void remove(const char* key);

        //Metadata
        size_t size()const;
        size_t capacity()const;
        //Presistence control
        void save_snapshot();
        void load_snapshot();
private:
            std::unique_ptr<IStorageBackend> backend_;
            std::unique_ptr<IPersistenceEngine> presistence_;
            mutable std::shared_mutex mutex_; //rd/wr lock
};

} //namespace inmemdb