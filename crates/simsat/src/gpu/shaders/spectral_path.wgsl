// Scalar transport CPU twin: spectral_path.rs. Host validates normalized phases,
// unit directions, SSA/albedo, roulette parameters and finite inputs.
// Backward directions use propagation convention: positive HG is forward.
// This module supplies sampling/material math and an all-order slab reference.
// It does NOT yet implement the actual 3D CPU scene traversal on the GPU.
const PATH_PI: f32 = 3.141592653589793;
struct PathPhase { rayleigh_weight:f32, gamma:f32, hg_g:vec4<f32>, hg_weight:vec4<f32> };
fn path_random_u32(state:ptr<function,u32>) -> u32 {
    *state=(*state)*747796405u+2891336453u;
    let word=(((*state)>>(((*state)>>28u)+4u))^(*state))*277803737u;
    return (word>>22u)^word;
}
fn path_uniform(state:ptr<function,u32>) -> f32 {
    return (f32(path_random_u32(state)>>9u)+0.5)/8388608.0;
}
fn path_direction_about_sine(axis:vec3<f32>,cosine:f32,sine:f32,azimuth:f32)->vec3<f32> {
    var helper=vec3<f32>(1.0,0.0,0.0);
    if(abs(axis.z)<0.9){helper=vec3<f32>(0.0,0.0,1.0);}
    let x=normalize(cross(helper,axis));let y=cross(axis,x);
    return normalize(cosine*axis+sine*(cos(azimuth)*x+sin(azimuth)*y));
}
fn path_direction_about(axis:vec3<f32>,cosine:f32,azimuth:f32)->vec3<f32> {
    return path_direction_about_sine(axis,cosine,sqrt(max(0.0,1.0-cosine*cosine)),azimuth);
}
fn path_phase_value(p:PathPhase,cosine:f32)->f32 {
    let mu=clamp(cosine,-1.0,1.0);
    var value=p.rayleigh_weight*3.0/(16.0*PATH_PI*(1.0+2.0*p.gamma))*
        ((1.0+3.0*p.gamma)+(1.0-p.gamma)*mu*mu);
    for(var k=0u;k<4u;k++){
        let g=p.hg_g[k];
        value+=p.hg_weight[k]*(1.0-g*g)/(4.0*PATH_PI*pow(1.0+g*g-2.0*g*mu,1.5));
    }
    return value;
}
fn path_phase_cosine(p:PathPhase,choose:f32,u:f32)->f32 {
    if(choose<p.rayleigh_weight){
        let isotropic=3.0*(1.0+3.0*p.gamma)/(4.0*(1.0+2.0*p.gamma));
        if(choose/p.rayleigh_weight<isotropic){return 2.0*u-1.0;}
        let x=2.0*u-1.0;return sign(x)*pow(abs(x),1.0/3.0);
    }
    var cumulative=p.rayleigh_weight;
    for(var k=0u;k<4u;k++){
        cumulative+=p.hg_weight[k];
        if(choose<cumulative){
            let g=p.hg_g[k];
            if(abs(g)<0.01){
                let x=2.0*u-1.0;
                let numerator=x+0.5*g*(x*x+3.0)+g*g*x+0.5*g*g*g*(x*x-1.0);
                let denominator=1.0+g*x;
                return clamp(numerator/(denominator*denominator),-1.0,1.0);
            }
            let q=(1.0-g*g)/(1.0-g+2.0*g*u);
            return clamp((1.0+g*g-q*q)/(2.0*g),-1.0,1.0);
        }
    }
    return 1.0;
}
fn path_phase_sample(p:PathPhase,direction:vec3<f32>,state:ptr<function,u32>)->vec3<f32>{
    let choose=path_uniform(state);let u=path_uniform(state);
    let cosine=path_phase_cosine(p,choose,u);let azimuth=2.0*PATH_PI*path_uniform(state);
    return path_direction_about(direction,cosine,azimuth);
}
fn path_fresnel_water(cosine:f32)->f32{
    let ci=clamp(cosine,0.0,1.0);let n=1.34;
    let ct=sqrt(1.0-(1.0-ci*ci)/(n*n));
    let rs=(ci-n*ct)/(ci+n*ct);let rp=(n*ci-ct)/(n*ci+ct);
    return clamp(0.5*(rs*rs+rp*rp),0.0,1.0);
}
// pi*BRDF; multiply by incidence cosine once to obtain direct rho_f.
fn path_cox_munk_kernel(source:vec3<f32>,view:vec3<f32>,normal:vec3<f32>,slope:f32)->f32{
    let s=normalize(source);let v=normalize(view);let up=normalize(normal);
    let mus=dot(s,up);let muv=dot(v,up);
    if(mus<=1e-4 || muv<=1e-4){return 0.0;}
    let half=s+v;let length_half=length(half);if(length_half<=1e-9){return 0.0;}
    let facet=half/length_half;let cb=clamp(dot(facet,up),1e-4,1.0);
    let cw=clamp(dot(s,facet),0.0,1.0);let cb2=cb*cb;
    let tan2=(1.0-cb2)/cb2;let mss=max(slope,1e-4);
    let density=exp(-tan2/mss)/(PATH_PI*mss);
    return max(0.0,PATH_PI*path_fresnel_water(cw)*density/(4.0*mus*muv*cb2*cb2));
}
// vec4(direction, BRDF*cos/PDF); w=0 means rejected/absorbed, never resampled.
fn path_lambertian_sample(view:vec3<f32>,normal:vec3<f32>,albedo:f32,state:ptr<function,u32>)->vec4<f32>{
    if(dot(view,normal)<=0.0 || albedo<=0.0){return vec4<f32>(0.0);}
    let cosine=sqrt(path_uniform(state));let azimuth=2.0*PATH_PI*path_uniform(state);
    return vec4<f32>(path_direction_about(normal,cosine,azimuth),albedo);
}
fn path_cox_munk_sample(view:vec3<f32>,normal:vec3<f32>,slope:f32,state:ptr<function,u32>)->vec4<f32>{
    if(dot(view,normal)<=1e-4){return vec4<f32>(0.0);}
    let mss=max(slope,1e-4);let tan2=-mss*log(path_uniform(state));
    let ch=1.0/sqrt(1.0+tan2);let azimuth=2.0*PATH_PI*path_uniform(state);
    // The equivalent slope-based sine avoids f32 cancellation near a flat facet.
    let facet=path_direction_about_sine(normal,ch,ch*sqrt(tan2),azimuth);let vh=dot(view,facet);
    if(vh<=0.0){return vec4<f32>(0.0);}
    let incoming=normalize(2.0*vh*facet-view);let mui=dot(incoming,normal);
    if(mui<=1e-4){return vec4<f32>(0.0);}
    let slope_pdf=exp(-tan2/mss)/(PATH_PI*mss);let pdf=slope_pdf/(ch*ch*ch)/(4.0*vh);
    let brdf=path_cox_munk_kernel(incoming,view,normal,mss)/PATH_PI;
    return vec4<f32>(incoming,brdf*mui/pdf);
}
// Returns rho_f total, first order, higher orders, event count. A negative event
// count signals the safety limit; callers MUST fail rather than use that path.
fn path_trace_slab(tau:f32,ssa:f32,phase:PathPhase,mu0:f32,muv:f32,azimuth:f32,
    albedo:f32,roulette_start:u32,roulette_threshold:f32,event_limit:u32,state:ptr<function,u32>)->vec4<f32>{
    let sinv=sqrt(max(0.0,1.0-muv*muv));
    var d=vec3<f32>(sinv*cos(azimuth),sinv*sin(azimuth),-muv);
    let sun=vec3<f32>(sqrt(max(0.0,1.0-mu0*mu0)),0.0,mu0);
    var z=tau;var weight=1.0;var first=0.0;var higher=0.0;var events=0u;
    for(var order=1u;order<=event_limit;order++){
        let sampled=-log(path_uniform(state));var boundary=3.402823e38;
        if(d.z>0.0){boundary=max(0.0,tau-z)/d.z;}
        else if(d.z<0.0){boundary=-max(0.0,z)/d.z;}
        var source=0.0;var bounce=1.0;var terminate=false;
        if(sampled<boundary){
            z+=sampled*d.z;weight*=ssa;
            if(weight==0.0){return vec4<f32>(PATH_PI*(first+higher),PATH_PI*first,PATH_PI*higher,f32(events));}
            source=path_phase_value(phase,dot(d,sun))*exp(-max(0.0,tau-z)/mu0);
            d=path_phase_sample(phase,d,state);
        }else if(d.z>0.0){
            return vec4<f32>(PATH_PI*(first+higher),PATH_PI*first,PATH_PI*higher,f32(events));
        }else{
            z=0.0;source=albedo*mu0/PATH_PI*exp(-tau/mu0);
            let next=path_lambertian_sample(-d,vec3<f32>(0.0,0.0,1.0),albedo,state);
            bounce=next.w;terminate=bounce==0.0;d=next.xyz;
        }
        events=order;
        if(order==1u){first+=weight*source;}else{higher+=weight*source;}
        if(terminate){return vec4<f32>(PATH_PI*(first+higher),PATH_PI*first,PATH_PI*higher,f32(events));}
        weight*=bounce;
        if(weight==0.0){return vec4<f32>(PATH_PI*(first+higher),PATH_PI*first,PATH_PI*higher,f32(events));}
        if(order>=roulette_start && weight<roulette_threshold){
            let survival=weight;
            if(path_uniform(state)>=survival){return vec4<f32>(PATH_PI*(first+higher),PATH_PI*first,PATH_PI*higher,f32(events));}
            weight/=survival;
        }
    }
    return vec4<f32>(0.0,0.0,0.0,-1.0);
}
