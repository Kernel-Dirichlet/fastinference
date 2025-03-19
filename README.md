# fastinference
Fast ML inference &amp; cross-platform Rust library

# LICENSE

AGPL-3.0 LICENSE 
   - subject to change 


## About

Fastinference is a rust library designed for TinyML use cases. 
This is an *inference only* library which hyper-accelerates inference time on 
devices which do not have the power capability to run GPUs and must rely on CPUs. 

Fastinference is cross-platform and the goal is to compile to various CPU architectures,
with a primary focus on x86 and ARM processors. Mid-term targets include PowerPC and RISC-Vchips and will include MIPS if this project generates sufficient interest. 

There will be support for embedded systems as well (which will not have a  memory allocatorfrom an OS) 
Actual articulate documentation and marketing to come at a later date. 



