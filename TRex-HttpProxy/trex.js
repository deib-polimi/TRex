var net = require('net');
var events = require('events');

//////////////////////////////////////////////////////////////////
//// Use the following environment variables to change defaults:
//// TREX_PORT             (default: 50254)
//// TREX_HOST             (default: 'localhost')
//////////////////////////////////////////////////////////////////

//////////////////////////////////////////////////////////////////
////// Auxiliary functions and constants
//////////////////////////////////////////////////////////////////

var pktType = {
    PUB: 0,
    SUB: 2,
    UNSUB:100
};

var valType = {
    INT: 0,
    FLOAT: 1,
    BOOL: 2,
    STRING: 3
};

function isInt(n) {
    return n === +n && n === (n|0);
}

function isFloat(n) {
    return n === +n && n !== (n|0);
}

function isBool(b) {
    return typeof b === "boolean";
}

function isString(s) {
    return typeof s === "string";
}

function getValType(val) {
    if(isInt(val)) return valType.INT;
    if(isFloat(val)) return valType.FLOAT;
    if(isBool(val)) return valType.BOOL;
    if(isString(val)) return valType.STRING;
}

//////////////////////////////////////////////////////////////////
//// Class Evt
//////////////////////////////////////////////////////////////////

function Evt(evtType, attr) {
    this.evtType = evtType;
    this.timeStamp = 0;
    if (typeof attr === 'undefined') this.attr = {};
    else this.attr = attr;
}

//////////////////////////////////////////////////////////////////
//// Class Sub
//////////////////////////////////////////////////////////////////

function Sub(evtType, constraints) {
    this.evtType = evtType;
    if (typeof constraints === 'undefined') this.constraints = [];
    else this.constraints = constraints;
}

Sub.prototype.addConstraint = function(attrName, operand, value) {
    this.constraints[this.constraints.length] = [attrName, operand, value];
}

Sub.op = {
    EQ: 0,
    LT: 1,
    GT: 2,
    NE: 3,
    IN: 4, 
    LE: 5,
    GE: 6
};

//////////////////////////////////////////////////////////////////
//// Marshalling functions
//////////////////////////////////////////////////////////////////

function subLen(sub) {
    var len = 4+4;  // evtType+numConstraints
    for(var i=0; i<sub.constraints.length; i++) {
	len += 4;   // contraint name length
	len += sub.constraints[i][0].length;
	len += 1+1; // Comparison operand + valType
	if(isInt(sub.constraints[i][2]) || isFloat(sub.constraints[i][2])) len += 4;
	else if(isBool(sub.constraints[i][2])) len += 1;
	else len += 4+sub.constraints[i][2].length;
    }
    return len;
}

function encodeSub(sub, buf, pos) { 
    buf.writeInt32BE(subLen(sub),pos); pos += 4;
    buf.writeInt32BE(sub.evtType,pos); pos += 4;  // evtType
    buf.writeInt32BE(sub.constraints.length,pos); pos += 4; // numConstr
    for(var i=0; i<sub.constraints.length; i++) {
	buf.writeInt32BE(sub.constraints[i][0].length, pos); pos += 4;
	buf.write(sub.constraints[i][0], pos, sub.constraints[i][0].length, "ascii"); pos += sub.constraints[i][0].length; 
	buf.writeInt8(sub.constraints[i][1], pos); pos += 1;
	buf.writeInt8(getValType(sub.constraints[i][2]), pos); pos += 1;
	if(isInt(sub.constraints[i][2])) { buf.writeInt32BE(sub.constraints[i][2], pos); pos += 4; }
	else if(isFloat(sub.constraints[i][2])) { buf.writeFloatBE(sub.constraints[i][2], pos); pos += 4; }
	else if(isBool(sub.constraints[i][2])) { buf.writeInt8(sub.constraints[i][2]?1:0, pos); pos += 1; }
	else { buf.writeInt32BE(sub.constraints[i][2].length, pos); pos += 4; buf.write(sub.constraints[i][2], pos, sub.constraints[i][2].length, "ascii"); pos += sub.constraints[i][2].length; }
    }
    return buf;
}

function getSubPkt(sub) {
    var buf = new Buffer(1+4+subLen(sub)); // pktType + bodyLen + body
    buf.writeInt8(pktType.SUB, 0);       // pktType
    encodeSub(sub, buf, 1);
    return buf;
}

function getUnsubPkt(sub) {
    var subPkt = getSubPkt(sub);
    var header = new Buffer(1+4); // pktType + subPktLen
    header.writeInt8(pktType.UNSUB, 0);  // pktType
    header.writeInt32BE(subPkt.length, 1);
    return Buffer.concat([header, subPkt]);
}

function evtLen(evt) {
    var len = 4+8+4;  // evtType+ts+numAttrs
    for(var i in evt.attr) {
	len += 4;   // attr name length
	len += i.length;  // attr name
	len += 1; // valType
	if(isInt(evt.attr[i]) || isFloat(evt.attr[i])) len += 4;
	else if(isBool(evt.attr[i])) len += 1;
	else len += 4+evt.attr[i].length;
    }
    return len;
}

function numAttr(evt) {
    var n=0;
    for(var i in evt.attr) n++;
    return n;
}

var POW32 = Math.pow(2,32);

function encodeEvt(evt, buf, pos) { 
    buf.writeInt32BE(evtLen(evt),pos); pos += 4;
    buf.writeInt32BE(evt.evtType,pos); pos += 4;  // evtType
    buf.writeInt32BE((evt.timeStamp-evt.timeStamp%POW32)/POW32,pos); pos += 4;  // high portion of timeStamp
    buf.writeUInt32BE(evt.timeStamp%POW32,pos); pos += 4;  // low portion of timeStamp
    buf.writeInt32BE(numAttr(evt),pos); pos += 4; // numattributes
    for(var i in evt.attr) {
	buf.writeInt32BE(i.length, pos); pos += 4;
	buf.write(i, pos, i.length, "ascii"); pos += i.length; 
	buf.writeInt8(getValType(evt.attr[i]), pos); pos += 1;
	if(isInt(evt.attr[i])) { buf.writeInt32BE(evt.attr[i], pos); pos += 4; }
	else if(isFloat(evt.attr[i])) { buf.writeFloatBE(evt.attr[i], pos); pos += 4; }
	else if(isBool(evt.attr[i])) { buf.writeInt8(evt.attr[i]?1:0, pos); pos += 1; }
	else { buf.writeInt32BE(evt.attr[i].length, pos); pos += 4; buf.write(evt.attr[i], pos, evt.attr[i].length, "ascii"); pos += evt.attr[i].length; }
    }
    return buf;
}

function parseEvtPkt(buf) {
    if(buf.readInt8(0) != pktType.PUB) {
	console.log("ERROR decoding packet");
	return ;
    } else return decodeEvt(buf, 1); 
}

function getEvtPkt(evt) {
    var buf = new Buffer(1+4+evtLen(evt)); // pktType + bodyLen + body
    buf.writeInt8(pktType.PUB, 0);
    encodeEvt(evt, buf, 1);
    return buf;
}

function decodeEvt(buf, pos) {
    pos += 4; // skip evt len
    var type = buf.readInt32BE(pos); pos+=4;
    var evt = new Evt(type);
    var ts = buf.readInt32BE(pos)*POW32; pos+=4;
    ts += buf.readUInt32BE(pos); pos+=4;
    evt.timeStamp = ts;
    var nAttr = buf.readInt32BE(pos); pos+=4;
    var attrName, attrNameLen, attrValType, attrVal, attrValLen;
    for(var i=0; i<nAttr; i++) {
	attrNameLen = buf.readInt32BE(pos); pos+=4;
	attrName = buf.toString("ascii", pos, pos+attrNameLen); pos+=attrNameLen;
	attrValType = buf.readInt8(pos); pos+=1;
	if(attrValType == valType.INT) {attrVal = buf.readInt32BE(pos); pos+=4;}
	else if(attrValType == valType.FLOAT) {attrVal = buf.readFloatBE(pos); pos+=4;}
	else if(attrValType == valType.BOOL) {attrVal = buf.readInt8(pos)==1; pos+=1;}
	else {attrValLen = buf.readInt32BE(pos); pos+=4; attrVal = buf.toString("ascii",pos, pos+attrValLen); pos+=attrValLen; }
	evt.attr[attrName] = attrVal;
    }
    return evt;
}

//////////////////////////////////////////////////////////////////
//// Class Connection
//////////////////////////////////////////////////////////////////

function Connection(port, host, connListener) {
    if (typeof port === 'undefined') this.port = process.env.TREX_PORT || 50254;
    else this.port = port;
    if (typeof host === 'undefined') this.host = process.env.TREX_HOST || '127.0.0.1';
    else this.host = host;
    var _events = new events.EventEmitter();
    this._events = _events;
    if (typeof connListener === 'undefined') this._sock = net.connect(this.port, this.host);
    else this._sock = net.connect(this.port, this.host, connListener);
    this._sock.on('data', function(data) {
	_events.emit('event', parseEvtPkt(data));
    });
}

Connection.prototype.close = function() {
    this._sock.end();
}

Connection.prototype.on = function(event, listener) {
    if(event === 'event') this._events.on(event, listener);
    else this._sock.on(event, listener);
}

Connection.prototype.addListener = Connection.prototype.on;

Connection.prototype.once = function(event, listener) {
    if(event === 'event') this._events.once(event, listener);
    else this._sock.once(event, listener);
}

Connection.prototype.removeListener = function(event, listener) {
    if(event === 'event') this._events.removeListener(event, listener);
    else this._sock.removeListener(event, listener);
}

Connection.prototype.removeAllListeners = function(event) {
    if(event === 'event') this._events.removeAllListeners(event);
    else this._sock.removeAllListeners(event);
}

Connection.prototype.listeners = function(event) {
    if(event === 'event') return this._events.listeners(event);
    else this._sock.listeners(event);
}

Connection.prototype.subscribe = function(sub) {
    this._sock.write(getSubPkt(sub));
}

Connection.prototype.unsubscribe = function(sub) {
    this._sock.write(getUnsubPkt(sub));
}

Connection.prototype.publish = function(evt) {
    this._sock.write(getEvtPkt(evt));
}

//////////////////////////////////////////////////////////////////
//// Function connect
//////////////////////////////////////////////////////////////////

function connect(port, host, connListener) {
    return new Connection(port, host, connListener);
}

exports.Sub = Sub;
exports.Evt = Evt;
exports.Connection = Connection;
exports.connect = connect;
