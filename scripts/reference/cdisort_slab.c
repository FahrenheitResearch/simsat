/* Independent scalar DISORT probe; external benchmark only. Does not modify or
   embed the reference solver in SimSat. Read normalized, one-layer cases from stdin.
   Input: id tau ssa kind g1 g2 weight gamma mu0 muv relaz_deg albedo nstr
   kind=0 depolarizing Rayleigh; kind=1 dual Henyey-Greenstein. */
#include "cdisort.h"
int main(void) {
    char line[1024];
    puts("id,nstr,L_over_E_sr1,rho_f,BRF,R,T_down,atmospheric_absorptance");
    while(fgets(line,sizeof(line),stdin)) {
        if(line[0]=='#' || line[0]=='\n')continue;
        int id,kind,nstr;double tau,ssa,g1,g2,w,gamma,mu0,muv,az,albedo;
        int n=sscanf(line,"%d %lf %lf %d %lf %lf %lf %lf %lf %lf %lf %lf %d",
            &id,&tau,&ssa,&kind,&g1,&g2,&w,&gamma,&mu0,&muv,&az,&albedo,&nstr);
        if(n!=13 || !isfinite(tau) || tau<=0.0 || !isfinite(ssa) || ssa<0.0 || ssa>1.0
           || (kind!=0 && kind!=1) || !isfinite(g1) || fabs(g1)>=1 || !isfinite(g2) || fabs(g2)>=1
           || !isfinite(w) || w<0 || w>1 || !isfinite(gamma) || gamma<0 || gamma>0.2
           || !isfinite(mu0) || mu0<=0 || mu0>1 || !isfinite(muv) || muv<=0 || muv>1
           || !isfinite(az) || az<0 || az>360 || !isfinite(albedo) || albedo<0 || albedo>1
           || nstr<4 || nstr>128 || nstr%2) {fputs("invalid case\n",stderr);return 2;}
        disort_state ds={0};disort_output out={0};
        ds.accur=0.0;ds.flag.ibcnd=GENERAL_BC;ds.flag.usrtau=TRUE;ds.flag.usrang=TRUE;
        ds.flag.lamber=TRUE;ds.flag.planck=FALSE;ds.flag.onlyfl=FALSE;ds.flag.quiet=TRUE;
        ds.flag.spher=FALSE;ds.flag.general_source=FALSE;ds.flag.output_uum=FALSE;
        ds.flag.intensity_correction=TRUE;ds.flag.old_intensity_correction=TRUE;ds.flag.brdf_type=BRDF_NONE;
        ds.nlyr=1;ds.nstr=nstr;ds.nphase=nstr;ds.nmom=512;ds.ntau=2;ds.numu=1;ds.nphi=1;
        c_disort_state_alloc(&ds);c_disort_out_alloc(&ds,&out);
        ds.dtauc[0]=tau;ds.ssalb[0]=ssa;ds.utau[0]=0;ds.utau[1]=tau;
        ds.umu[0]=muv;ds.phi[0]=az;ds.bc.fbeam=1.0;ds.bc.umu0=mu0;ds.bc.phi0=0;
        ds.bc.fisot=0;ds.bc.fluor=0;ds.bc.albedo=albedo;
        for(int k=0;k<=ds.nmom;k++) {
            if(kind==0) ds.pmom[k]=(k==0)?1.0:((k==2)?(1-gamma)/(10*(1+2*gamma)):0.0);
            else ds.pmom[k]=w*pow(g1,k)+(1-w)*pow(g2,k);
        }
        c_disort(&ds,&out);
        double L=out.uu[0],R=out.rad[0].flup/mu0,T=(out.rad[1].rfldir+out.rad[1].rfldn)/mu0;
        if(!isfinite(L)||!isfinite(R)||!isfinite(T)){fputs("non-finite reference\n",stderr);return 3;}
        printf("%d,%d,%.17g,%.17g,%.17g,%.17g,%.17g,%.17g\n",id,nstr,L,M_PI*L,M_PI*L/mu0,R,T,1-R-(1-albedo)*T);
        c_disort_out_free(&ds,&out);c_disort_state_free(&ds);
    }
    return 0;
}
