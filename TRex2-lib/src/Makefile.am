.cu.o:
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)
.cu.lo:
	$(NVCC) -o $@ -c $< $(NVCCFLAGS)

Unmarshaller.${OBJEXT} : CXXFLAGS = -O0
	
lib_LTLIBRARIES = libTRex2.la
libTRex2_la_SOURCES = Common/CompositeEventTemplate.cc Common/Funs.cc Common/OpTree.cc Common/TimeMs.cc Common/Timer.cc Common/Consts.h \
	Engine/AbstractConstraintIndex.cc Engine/BoolConstraintIndex.cc Engine/CompositeEventGenerator.cc Engine/FloatConstraintIndex.cc Engine/IndexingTable.cc Engine/IntConstraintIndex.cc Engine/NoConstraintIndex.cc Engine/Stack.cc Engine/StacksRule.cc Engine/StringConstraintIndex.cc Engine/TRexEngine.cc \
	Marshalling/Marshaller.cc Marshalling/Unmarshaller.cc \
	Packets/AdvPkt.cc Packets/JoinPkt.cc Packets/PubPkt.cc Packets/RulePkt.cc Packets/RulePktValueReference.cc Packets/SubPkt.cc Packets/StaticValueReference.cc
	
engineheadersdir = $(includedir)/TRex2/Engine
engineheaders_HEADERS = Engine/TRexEngine.h Engine/AbstractConstraintIndex.h Engine/BoolConstraintIndex.h Engine/CompositeEventGenerator.h Engine/FloatConstraintIndex.h Engine/IndexingTable.h Engine/IntConstraintIndex.h Engine/NoConstraintIndex.h Engine/Stack.h Engine/StacksRule.h Engine/StringConstraintIndex.h Engine/Engine.h Engine/IndexingTableCommon.h Engine/ResultListener.h
	

commonheadersdir = $(includedir)/TRex2/Common
commonheaders_HEADERS = Common/Consts.h Common/CompositeEventTemplate.h Common/Funs.h Common/OpTree.h Common/TimeMs.h Common/Timer.h Common/Structures.h Common/OpValueReference.h

marshheadersdir = $(includedir)/TRex2/Marshalling
marshheaders_HEADERS = Marshalling/Marshaller.h Marshalling/Unmarshaller.h

pktsheadersdir = $(includedir)/TRex2/Packets
pktsheaders_HEADERS = Packets/PubPkt.h Packets/RulePkt.h Packets/RulePktValueReference.h Packets/StaticValueReference.h Packets/AdvPkt.h Packets/JoinPkt.h Packets/SubPkt.h
	
if USE_GPU
gengineheadersdir = $(includedir)/TRex2/Engine/GPU
gengineheaders_HEADERS = Engine/GPU/GPUEngine.h Engine/GPU/GPUProcessorIf.h Engine/GPU/GPUProcessor.h Engine/GPU/MemoryManager.h Engine/GPU/CudaKernels.h
EXTRA_DIST = Engine/GPU/MemoryManager.cu Engine/GPU/CudaKernels.cu
libTRex2_la_SOURCES += Engine/GPU/GPUEngine.cc Engine/GPU/GPUProcessor.cc
libTRex2_la_LIBADD = Engine/GPU/MemoryManager.o Engine/GPU/CudaKernels.o
endif
