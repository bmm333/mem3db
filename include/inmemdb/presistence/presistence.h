#pragma once
#include <string>

namespace inmemdb{
    class IStorageBackend;
    enum class CommandType{
        PUT,
        DELETE
    };
    struct Command{
        CommandType type;
        char key[64];
        char value[256];
    };
    class IPersistenceEngine{
        public:
            virtual ~IPersistenceEngine()=default;
            //Write ahead logging
            virtual void log_command(const Command& cmd)=0;
            virtual void flush()=0;
            //Snapshots
            virtual void save_snapshot(const IStorageBackend& backend)=0
            virtual void load_snapshot(IStorageBackend& backend)=0;
            //Recovery
            virtual void reply_wal(IStorageBackend& backend)=0;
            //lifecycle
            virtual void start_background_sync()=0;
            virtual void stop_background_sync()=0;
    };
} //namespace inmemdb