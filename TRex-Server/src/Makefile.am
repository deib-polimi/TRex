LDFLAGS_BOOST = \
    $(BOOST_THREAD_LDFLAGS) \
    $(BOOST_SYSTEM_LDFLAGS) \
    $(BOOST_LOG_LDFLAGS) \
    $(BOOST_LOG_SETUP_LDFLAGS) \
    $(BOOST_ARRAY_LDFLAGS)

LIBS_BOOST = \
    $(BOOST_THREAD_LIBS) \
    $(BOOST_SYSTEM_LIBS) \
    $(BOOST_LOG_LIBS) \
    $(BOOST_LOG_SETUP_LIBS) \
    $(BOOST_ARRAY_LIBS)

AM_CPPFLAGS = $(BOOST_CPPFLAGS)
AM_LDFLAGS = $(LDFLAGS_BOOST)
LIBS += $(LIBS_BOOST)

bin_PROGRAMS = TRexServer
TRexServer_SOURCES = Connection/Connection.cpp Connection/ConnectionProxy.cpp Connection/PubPktListener.cpp Connection/RequestHandler.cpp Connection/SubscriptionTable.cpp \
	Packet/BufferedPacketUnmarshaller.cpp Packet/PacketMarshaller.cpp Packet/PingPkt.cpp Packet/UnSubPkt.cpp \
	Server/SOEPServer.cpp \
	Test/RuleR0.cpp Test/RuleR1.cpp Test/TestResultListener.cpp \
	Util/TRexUtils.cpp Util/Logging.cpp \
	Main.cpp
	
TRexServer_CPPFLAGS = $(BOOST_CPPFLAGS) -DBOOST_LOG_DYN_LINK
TRexServer_LDFLAGS = $(BOOST_THREAD_LDFLAGS) $(BOOST_SYSTEM_LDFLAGS) $(BOOST_LOG_LDFLAGS) $(BOOST_LOG_SETUP_LDFLAGS)
TRexServer_LIBS = $(BOOST_THREAD_LIBS) $(BOOST_SYSTEM_LIBS) $(BOOST_LOG_LIBS) $(BOOST_LOG_SETUP_LIBS)
