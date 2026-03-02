import torch
from .memory_efficient.frugal import AdamW, GaloreAdamW, CoordAdamW, BlockAdamW, SGD, GaloreSGD, CoordSGD, BlockSGD, Lion, GaloreLion, CoordLion, BlockLion, Adalayer, GaloreAdalayer, CoordAdalayer, BlockAdalayer
from .memory_efficient.fira import FiraAdamW
from .memory_efficient.galore import GaLoreAdafactor, AdaMeM
from .memory_efficient.apollo import APOLLOAdamW
from .memory_efficient.ldadam import LDAdamW
from .sota_opt import AdEMAMix, dion, Adan, ADOPT, SOAP, MARS, MARS_M


def get_optimizer(param_groups, args, model=None, qargs=None):
    
    optimizer_name = args.opt.lower()
    assert optimizer_name != "badam" or model is not None, "BAdam requires model for the initialization."
    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)#, foreach=False, fused=False)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)#, foreach=False, fused=False)
    elif optimizer_name == "ademamix":
        optimizer = AdEMAMix(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2, args.ademamix_beta3),
            alpha=args.ademamix_alpha,
            beta3_warmup=args.ademamix_beta3_warmup_steps,
            alpha_warmup=args.ademamix_alpha_warmup_steps,
            eps=args.eps,
            weight_decay=args.weight_decay,
        )
    elif optimizer_name == "muon":
        for group in param_groups:
            if not group.get("is_proj_params", False):
                if args.muon_1d_backup == "adan":
                    group["algorithm"] = "adan"
                else:
                    group["algorithm"] = "adamw"
        optimizer = dion.Muon(
            param_groups,
            distributed_mesh=model.process_group,
            lr=args.lr,
            mu=args.momentum,
            betas=(args.beta1, args.beta2),
            adan_betas=(args.adan_beta1, args.adan_beta2, args.adan_beta3),
            weight_decay=args.weight_decay,
            epsilon=args.eps,
            nesterov=args.nesterov,
            adjust_lr=args.muon_adjust_lr,
            newton_schulz_func_name=args.newton_schulz_func,
            pre_orth_update_name=args.muon_pre_orth_update,
            muon_ns_steps=args.muon_ns_steps,
            headwise=args.muon_headwise,
            adamw_lr_scale=args.muon_adamw_lr_scale,
            dampening=args.dampening
        )
    elif optimizer_name == "normuon":
        for group in param_groups:
            if not group.get("is_proj_params", False):
                group["algorithm"] = "adamw"
        optimizer = dion.NorMuon(
            param_groups,
            distributed_mesh=model.process_group,
            lr=args.lr,
            mu=args.momentum,
            muon_beta2=args.muon_beta2,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            epsilon=args.eps,
            nesterov=args.nesterov,
            adjust_lr=args.muon_adjust_lr,
            use_adamuon=False,
        )
    elif optimizer_name == "adamuon":
        for group in param_groups:
            if not group.get("is_proj_params", False):
                group["algorithm"] = "adamw"
        optimizer = dion.NorMuon(
            param_groups,
            distributed_mesh=model.process_group,
            lr=args.lr,
            mu=args.momentum,
            muon_beta2=args.muon_beta2,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
            epsilon=args.eps,
            nesterov=args.nesterov,
            adjust_lr=args.muon_adjust_lr,
            use_adamuon=True,
        )
    elif optimizer_name == "adan":
        optimizer = Adan.Adan(
            param_groups,
            lr=args.lr,
            betas=(args.adan_beta1, args.adan_beta2, args.adan_beta3),
            eps=args.eps,
            weight_decay=args.weight_decay,
            max_grad_norm=args.adan_max_grad_norm,
            no_prox=args.adan_no_prox,
            # foreach = False,
            # fused = False,
        )
    elif optimizer_name == "nadam":
        optimizer = torch.optim.NAdam(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            momentum_decay=args.nadam_momentum_decay,
            decoupled_weight_decay=True,
            # foreach = False,
            # fused = False,
        )
    elif optimizer_name == "adopt":
        optimizer = ADOPT(
            param_groups,
            lr=args.lr,
            betas=(args.adopt_beta1, args.adopt_beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            decouple=args.adopt_decouple
        )
    elif optimizer_name == "soap":
        optimizer = SOAP(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            shampoo_beta=args.shampoo_beta,
            eps=args.eps,
            weight_decay=args.weight_decay,
            precondition_frequency=args.update_gap,
            precondition_embed_debed=args.soap_precondition_embed_debed,
        )
    elif optimizer_name == "mars":
        optimizer = MARS(
            param_groups,
            lr=args.lr,
            betas=(args.mars_beta1, args.mars_beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,
            # amsgrad=False,
            gamma=args.mars_gamma,
            # is_approx=True,
            mars_type=args.mars_type,

            # optimize_1d=False,
            lr_1d=args.lr,
            # betas_1d=(args.beta1, args.beta2),
            weight_decay_1d=args.weight_decay,
        )
    elif optimizer_name == "mars_m":
        muon_params = []
        adamw_params = []
        for group in param_groups:
            if group.get("is_proj_params", False):
                muon_params.extend(group["params"])
            else:
                adamw_params.extend(group["params"])
        optimizer = MARS_M(
            lr=args.lr,
            wd=args.weight_decay,
            muon_params=muon_params,
            momentum=args.momentum,
            ns_steps=args.muon_ns_steps,
            adamw_params=adamw_params,
            adamw_betas=(args.beta1, args.beta2),
            adamw_eps=args.eps,
            gamma=args.mars_gamma,
            # clip_c=False,
            # is_approx=True,
        )
    elif "apollo" in optimizer_name:
        for group in param_groups:
            if group.get("is_proj_params", False):
                if "mini" in optimizer_name:
                    group["rank"] = args.rank
                else:
                    group["rank"] = int(args.density * args.hidden_size)
                print("\n"*3)
                print("-"*30)
                print(f"APOLLO Rank: {group['rank']}")
                print("-"*30)
                print("\n"*3)
                group["update_proj_gap"] = args.update_gap
                group["scale"] = args.apollo_scale
                group["proj"] = args.apollo_proj
                group["scale_type"] = args.apollo_scale_type

                group["proj_type"] = args.proj_side
        optimizer = APOLLOAdamW(param_groups, lr=args.lr, weight_decay=args.weight_decay, scale_front=args.apollo_scale_front)
    elif optimizer_name == "ldadam":
        for group in param_groups:
            if group.get("is_proj_params", False):
                group["enable_lowrank"] = True
                group["rank"] = int(args.density * args.hidden_size)
                print("\n"*3)
                print("-"*30)
                print(f"LDAdam Rank: {group['rank']}")
                print("-"*30)
                print("\n"*3)
                group["rho"] = args.ldadam_rho
                group["proj_method"] = args.ldadam_proj_method
                group["error_feedback"] = args.ldadam_error_feedback

                group["proj_type"] = args.proj_side
            else:
                group["enable_lowrank"] = False
        optimizer = LDAdamW(param_groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
    elif optimizer_name == "fira_adamw":
        for group in param_groups:
            if group.get("is_proj_params", False):
                group["rank"] = int(args.density * args.hidden_size)
                print("\n"*3)
                print("-"*30)
                print(f"Fira Rank: {group['rank']}")
                print("-"*30)
                print("\n"*3)
                group["update_proj_gap"] = args.update_gap
                group["alpha"] = args.fira_alpha

                group["proj_side"] = args.proj_side
                group["proj_type"] = args.proj_type

                group["reset_statistics"] = args.reset_statistics
        optimizer = FiraAdamW(param_groups, lr=args.lr, betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay)
    elif optimizer_name == "galore_adafactor":
        for group in param_groups:
            if group.get("is_proj_params", False):
                group["rank"] = int(args.density * args.hidden_size)
                print("\n"*3)
                print("-"*30)
                print(f"Galore Adafactor Rank: {group['rank']}")
                print("-"*30)
                print("\n"*3)
                group["update_proj_gap"] = args.update_gap
                group["scale"] = args.proj_params_lr_scale
                group["proj_type"] = args.proj_side
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    elif optimizer_name == "adamem":
        for group in param_groups:
            if group.get("is_proj_params", False):
                group["rank"] = int(args.density * args.hidden_size)
                print("\n"*3)
                print("-"*30)
                print(f"AdaMeM Adafactor Rank: {group['rank']}")
                print("-"*30)
                print("\n"*3)
                group["update_proj_gap"] = args.update_gap
                group["scale"] = args.proj_params_lr_scale
                group["proj_type"] = args.proj_side

                group["use_adamem"] = True
                group["adamem_type"] = args.adamem_type
                group["adamem_rms"] = args.adamem_rms
                group["adamem_relative_lr"] = args.adamem_relative_lr
                group["adamem_reduce_op"] = args.adamem_reduce_op
        optimizer = GaLoreAdafactor(
            param_groups,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    elif optimizer_name == "adamem_orig":
        for group in param_groups:
            if group.get("is_proj_params", False):
                group["rank"] = int(args.density * args.hidden_size)
                print("\n"*3)
                print("-"*30)
                print(f"AdaMeM Adafactor Rank: {group['rank']}")
                print("-"*30)
                print("\n"*3)
                group["update_proj_gap"] = args.update_gap
                group["scale"] = args.proj_params_lr_scale
                group["proj_type"] = args.proj_side
                group["adamem_reduce_op"] = args.adamem_reduce_op

        optimizer = AdaMeM(
            param_groups,
            lr=args.lr,
            betas=(args.beta1, args.beta2),
            eps=args.eps,
            weight_decay=args.weight_decay,

            adamem_relative_lr=args.adamem_relative_lr,
            use_momentum_to_update_variance=args.use_momentum_to_update_variance,
        )

    elif optimizer_name == "galore_adamw":
        # redefine way to call galore_adamw
        optimizer = GaloreAdamW(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # galore specific
            proj_side=args.proj_side,
            proj_type=args.proj_type,
            # adam specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif optimizer_name == "coord_adamw":
        optimizer = CoordAdamW(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            coord_choice=args.coord_choice,
            # adam specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif optimizer_name == "block_adamw":
        optimizer = BlockAdamW(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            block_order=args.block_order,
            # adam specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif optimizer_name == "adalayer":
        for group in param_groups:
            group["sqrt_numel"] = args.sqrt_numel
        optimizer = Adalayer(param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif optimizer_name == "block_adalayer":
        for group in param_groups:
            group["sqrt_numel"] = args.sqrt_numel
        optimizer = BlockAdalayer(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            block_order=args.block_order,
            # adalayer specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps)
    elif optimizer_name == "lion":
        optimizer = Lion(param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == "galore_lion":
        optimizer = GaloreLion(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # galore specific
            proj_side=args.proj_side,
            proj_type=args.proj_type,
            # lion specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == "coord_lion":
        optimizer = CoordLion(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            coord_choice=args.coord_choice,
            # lion specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay)
    elif optimizer_name == "block_lion":
        optimizer = BlockLion(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            block_order=args.block_order,
            # lion specific
            betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay)
    # implement sgd
    elif optimizer_name == "sgd":
        optimizer = SGD(param_groups, lr=args.lr, momentum=args.beta1, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov, sign_update=args.sgd_sign_update)
    elif optimizer_name == "galore_sgd":
        optimizer = GaloreSGD(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # galore specific
            proj_side=args.proj_side,
            proj_type=args.proj_type,
            # sgd specific
            lr=args.lr, momentum=args.beta1, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov, sign_update=args.sgd_sign_update)
    elif optimizer_name == "coord_sgd":
        optimizer = CoordSGD(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # galore specific
            coord_choice=args.coord_choice,
            # sgd specific
            lr=args.lr, momentum=args.beta1, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov, sign_update=args.sgd_sign_update)
    elif optimizer_name == "block_sgd":
        optimizer = BlockSGD(
            param_groups,
            proj_params_lr_scale=args.proj_params_lr_scale,
            update_gap = args.update_gap,
            density=args.density,
            reset_statistics=args.reset_statistics,
            inactive_update_rule=args.inactive_update_rule,
            inactive_lr_scale=args.inactive_lr_scale,
            # coord specific
            block_order=args.block_order,
            # lion specific
            lr=args.lr, momentum=args.beta1, dampening=args.dampening, weight_decay=args.weight_decay, nesterov=args.nesterov, sign_update=args.sgd_sign_update)
    elif optimizer_name == "badam":
        raise NotImplementedError
        from badam import BlockOptimizer as BAdamBlockOptimizer
        original_optimizer = torch.optim.Adam(param_groups, betas=(args.beta1, args.beta2), lr=args.lr, weight_decay=args.weight_decay, eps=args.eps, foreach=False, fused=False)
        num_layers = len(model.model.layers)
        block_size = int(args.density * num_layers)
        if num_layers % block_size:
            raise ValueError("Incorrect density - can't split layers in equal parts.")
        block_prefix_list = []
        for block_start in range(0, num_layers, block_size):
            cur_block_prefix_list = []
            for idx in range(block_start, block_start + block_size):
                cur_block_prefix_list.append(f"model.layers.{idx}.self_attn.")
                cur_block_prefix_list.append(f"model.layers.{idx}.mlp")
            block_prefix_list.append(cur_block_prefix_list)
        optimizer = BAdamBlockOptimizer(
            base_optimizer=original_optimizer,
            named_parameters_list=list(model.named_parameters()),
            block_prefix_list=block_prefix_list,
            active_modules=[name for name, _ in model.named_parameters() if "embed" in name or "norm" in name or "head" in name],
            switch_block_every=args.update_gap,
            switch_mode=args.block_order,
            verbose=2,
        )
    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")
    return optimizer
