var express = require('express');
var bodyParser = require('body-parser');
var trex = require('./trex.js');


//////////////////////////////////////////////////////////////////
//// Use the following environment variables to change defaults:
//// TREX_PROXY_PORT       (default: 8888)
//// TREX_PROXY_CONN_LEASE (default: 600)
//// TREX_PORT             (default: 50254)
//// TREX_HOST             (default: 'localhost')
//////////////////////////////////////////////////////////////////


var leaseTime = process.env.TREX_PROXY_CONN_LEASE || 600; // timeout before removing connections, in seconds

var connections = [];
var connectionsTS = [];
var events = [];

function generateUUID() {
    var d = Date.now();
    var uuid = 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        var r = (d + Math.random()*16)%16 | 0;
        d = Math.floor(d/16);
        return (c=='x' ? r : (r&0x3|0x8)).toString(16);
    });
    return uuid;
};

function getConnection(connID) {
    if(typeof connID === 'undefined') {
	connID = generateUUID();
	var conn;
	try {
	    conn = trex.connect(process.env.TREX_PORT || 50254, process.env.TREX_HOST || 'localhost', function() { //'connect' listener
		console.log('Client '+connID+' connected');
	    });
	} catch(err) {
	    console.log('Error connecting to the TRex server :'+err);
	    return;
	}	
	conn.on('event', function(evt) {
	    var stringifiedEvt = JSON.stringify(evt);
	    console.log('Event for client: "'+connID+'": '+stringifiedEvt);
	    events[connID].push(evt);
	    connectionsTS[connID]=Date.now();
	});
	
	conn.on('error', function(err) {
	    console.log(err+' on connection "'+connID+'".');
	});

	conn.on('end', function() {
	    console.log('Connection for client "'+connID+'" ended');
	    delete connectionsTS[connID];
	    delete connections[connID];
	    delete events[connID];
	});
		
	conn.on('close', function() {
	    console.log('Connection for client "'+connID+'" closed');
	    delete connectionsTS[connID];
	    delete connections[connID];
	    delete events[connID];
	});

	connections[connID] = conn;
	connectionsTS[connID] = Date.now();
	events[connID] = [];
	return connID;
    } else {
	connectionsTS[connID] = Date.now();
	return connections[connID];
    }
}

function cleanUp() {
    var now = Date.now();
    for(var c in connectionsTS) {
	if((now-connectionsTS[c])>leaseTime*1000) {
	    console.log('Connection for client "'+c+'" expired');
	    connections[c].close();
	}
    }
}

setInterval(cleanUp, 5000);

var app = express();

console.log('Serving static content from '+__dirname);
app.use(express.static(__dirname));

app.use(bodyParser.json()); // for parsing application/json
app.use(bodyParser.urlencoded({ extended: true })); // for parsing application/x-www-form-urlencoded

app.get('/connections', function(req, res) {
    console.log("GET /connections");
    res.json(getConnection());
});

app.post('/subscriptions/:connID', function(req, res) {
    console.log("POST /subscriptions");
    console.log('\tclient '+req.params.connID+' sent "'+JSON.stringify(req.body)+'" data');
    var conn = getConnection(req.params.connID);
    if(typeof conn === 'undefined') {
	res.status(404);
	return res.send('Error 404: No connection ID found');
    }
    conn.subscribe(req.body);
    res.status(200);
    res.json(true);
});

app.delete('/subscriptions/:connID', function(req, res) {
    console.log("DELETE /subscriptions");
    console.log('\tclient '+req.params.connID+' sent "'+JSON.stringify(req.body)+'" data');
    var conn = getConnection(req.params.connID);
    if(typeof conn === 'undefined') {
	res.status(404);
	return res.send('Error 404: No connection ID found');
    }
    conn.unsubscribe(req.body);
    res.status(200);
    res.json(true);
});

app.get('/events/:connID', function(req, res) {
    console.log("GET /events");
    console.log('\tclient '+req.params.connID+' sent "'+JSON.stringify(req.body)+'" data');
    var conn = getConnection(req.params.connID);
    if(typeof conn === 'undefined') {
	res.status(404);
	return res.send('Error 404: No connection ID found');
    }
    if(typeof events[req.params.connID] === 'undefined') {
	res.status(404);
	return res.send('Error 404: No connection ID found');
    }
    if(events[req.params.connID].length==0) {
	res.status(204);
	return res.send();
    }
    res.status(200);
    res.json(events[req.params.connID].shift());
});

app.post('/events/:connID', function(req, res) {
    console.log("POST /events");
    console.log('\tclient '+req.params.connID+' sent "'+JSON.stringify(req.body)+'" data');
    var conn = getConnection(req.params.connID);
    if(typeof conn === 'undefined') {
	res.status(404);
	return res.send('Error 404: No connection ID found');
    }
    conn.publish(req.body);
    res.status(200);
    res.json(true);
});


app.listen(process.env.TREX_PROXY_PORT || 8888);
