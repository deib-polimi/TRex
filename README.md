T-Rex
=====

T-Rex is a general purpose Complex Event Processing (CEP) Middleware designed to support an expressive language for rule definition while offering efficient processing mechanisms.

With the T-Rex tools users can create TESLA rules, publish events and subscribe to event notifications.

TESLA
=====

TESLA is a powerful yet simple complex event definition language.

TESLA includes operators to express:

- Content-based event filtering
- Customizable event selection and consumption policies
- Event sequences with timing constraints
- Negations (time-based and interval-based)
- Parameters
- Aggregates

Further details on TESLA can be found in the following paper: 
TESLA: a Formally Defined Event Specification Language, G. Cugola, A. Margara. In Proceedings of 4th ACM International Conference On Distributed Event-Based Systems (DEBS 2010). Cambridge, United Kingdom. July 12 - 15, 2010
> http://www.inf.usi.ch/postdoc/margara/papers/debs10.pdf

Architecture
============

The T-Rex software suite consists of 3 components: a C++/CUDA highly optimized engine, a C++ server and a Java client.

Further details on T-Rex algorithms and implementation can be found in the following paper: 
Complex Event Processing with T-Rex, G. Cugola, A. Margara. In Journal of Systems and Software. Volume 85 - Issue 8. August 2012.
> http://home.deib.polimi.it/cugola/Papers/trex.pdf

More on the GPU implementation can be found in the following thesis: 
GTrex : implementing a CEP engine for the TESLA language on GPU, Daniele Rogora.
> https://www.politesi.polimi.it/bitstream/10589/88428/1/Tesi.pdf

The client includes the code to actually perform the communication between the client itself and the server as well as a TESLA rule parser.

More on the specific syntax used for the parser can be found in the PDF document **TESLA_parser_intro** in this repository.

TRex2-Lib
=========

This is the core of the project, containing both the standard and the CUDA powered engine. 

First of all build the configure script:

    $ autoreconf --install

To configure:

    $ ./configure
    
To build also the CUDA engine:

    $ ./configure --enable-cuda
    
Build the binary library:

    $ make
    
To install the library and the development headers:

    $ make install
        

TRex-Server
===========

This is a server built with the TRex2 library that handles the communication between the engine and the clients.

First of all build the configure script:

    $ autoreconf --install

To configure:

    $ ./configure
    
Build the binary:

    $ make
    
To install the library and the development headers:

    $ make install
    
To run the server with the standard engine:

    $ TRexServer
    
To run the server with the CUDA engine:

    $ TRexServer -gpu

TRex-Java-client
================

This project contains both a Java library to simply and efficiently take advantage of the T-Rex server and an example command line client developed with the library.

To build the example command line client:

    $ ant jars
    
Run with:

    $ java -jar TRex-client.jar 
    
Other options can be seen with:

    $ ant -p

TRex-HttpProxy
================

This project contains a very simple proxy written in node.js, which converts REST calls into calls to the T-Rex server.

The proxy can be launched through the command:

    $ nodejs trex-proxy.js

waits connections on port 8888, and connects to the T-Rex server running on localhost:50254 (the default). Change file "handlers.js" to connect to a different server.

The proxy exports four main REST services: one to connect, one to publish events, one to subscribe to events, and one to get received events (which it stores, on behalf of clients). The first two are POST requests, the latter is a GET request.

Clients are identified through an UUID they receive at connection time.

File "testjs.html" (served through the URL: http://proxy.machine:8888/testjs.html) provides a starting example of usage from Javascript. The javascript client-side library used by such an example can be found in file trex-client.js.



Contacts
========

Gianpaolo Cugola

> http://home.deib.polimi.it/cugola

Alessandro Margara

> http://www.inf.usi.ch/postdoc/margara/index.html

Daniele Rogora 