require 'ffi';

module Hello
  extend FFI::Library
  ffi_lib 'target/release/libembed.dylib'
  attach_function :process, [], :void
end
t1 = Time.now
# processing...
Hello.process
t2 = Time.now
delta = t2 - t1

puts delta
