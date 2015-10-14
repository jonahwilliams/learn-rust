var ffi = require('ffi');
var lib = ffi.Library('target/release/libembed', {
    process: ['void', []]
});

console.time();
lib.process();
console.timeEnd();
