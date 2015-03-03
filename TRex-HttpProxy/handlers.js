var fs = require("fs");
var querystring = require("querystring");
var trex = require('./trex.js');

var LEASE_TIME = 600000; // 5 minutes
var TREX_HOST = "localhost"
var TREX_PORT = 50254

var connection = [];
var connectionTS = [];
var events = [];

function getConnection(client) {
    if(typeof connection[client] === 'undefined') {
	var conn = trex.connect(TREX_PORT, TREX_HOST, function() { //'connect' listener
	    console.log('Connected');
	});
	
	conn.on('event', function(evt) {
	    var stringifiedEvt = JSON.stringify(evt);
	    console.log('Event for client: "'+client+'": '+stringifiedEvt);
	    events[client].push(evt);
	    connectionTS[client]=Date.now();
	});
	
	conn.on('end', function() {
	    console.log('Connection for client "'+client+'" ended');
	    delete connectionTS[client];
	    delete connection[client];
	    delete events[client];
	});
	
	
	conn.on('close', function() {
	    console.log('Connection for client "'+client+'" closed');
	    delete connectionTS[client];
	    delete connection[client];
	    delete events[client];
	});

	connection[client] = conn;
	connectionTS[client] = Date.now();
	events[client] = [];
    }
    return connection[client];
}

function cleanUp() {
    var now = Date.now();
    for(var c in connectionTS) {
	if((now-connectionTS[c])>LEASE_TIME) {
	    console.log('Connection for client "'+c+'" expired');
	    connection[c].close();
	}
    }
}

setInterval(cleanUp, 5000);

function publish(query, postData, response) {
    console.log("Request handler 'publish' was called with '"+query+"' query, and '"+postData+"' data.");
    var clientID = querystring.parse(query).clientID;
    var evt = JSON.parse(querystring.parse(postData).event);
    response.writeHead(200, {"Content-Type": "text/plain"});
    // response.write("Event: '"+JSON.stringify(evt)+"' published on behalf of client "+clientID);
    response.end();
    console.log("Client "+clientID+" publishing event: "+JSON.stringify(evt)+"'");
    getConnection(clientID).publish(evt);
}

function subscribe(query, postData, response) {
    console.log("Request handler 'subscribe' was called with '"+query+"' query, and '"+postData+"' data.");
    var clientID = querystring.parse(query).clientID;
    var sub = JSON.parse(querystring.parse(postData).subscription);
    response.writeHead(200, {"Content-Type": "text/plain"});
    // response.write("Subscription: '"+JSON.stringify(sub)+"' done on behalf of client "+clientID);
    response.end();
    console.log("Client "+clientID+" subscribing to: '"+JSON.stringify(sub)+"'");
    getConnection(clientID).subscribe(sub);
}

function getevent(query, postData, response) {
    console.log("Request handler 'getevent' was called with '"+query+"' query, and '"+postData+"' data.");
    var clientID = querystring.parse(query).clientID;
    if(!(typeof events[clientID] === 'undefined') && events[clientID].length!=0) { 
	response.writeHead(200, {"Content-Type": "application/json"});
	response.write(JSON.stringify(events[clientID].shift()));
    }
    response.end();
    if(!(typeof connectionTS[clientID] === 'undefined')) connectionTS[clientID]=Date.now();
}

function test(query, postData, response) {
    console.log("Request handler 'test' was called with '"+query+"' query, and '"+postData+"' data.");
    uuid=trex.generateUUID();
    fs.readFile('./test.html', {encoding: 'utf-8'}, function(err,data){
	if (!err){
	    response.writeHead(200);
	    response.write(data.replace(/\$UUID/g,uuid));
	    response.end();
	} else {
	    console.log(err);
	    response.writeHead(404, {"Content-Type": "text/plain"});
	    response.write("404 Not found");
	    response.end();
	}
    });
}


var handle = {
    "/publish": publish,
    "/subscribe": subscribe,
    "/getevent": getevent,
    "/test": test
}


exports.handle = handle;

