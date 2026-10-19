package org.playerone.bridge.mixin;

import net.minecraft.client.Minecraft;
import net.minecraft.client.MouseHandler;
import org.playerone.bridge.InputApplier;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.injection.At;
import org.spongepowered.asm.mixin.injection.Inject;
import org.spongepowered.asm.mixin.injection.Redirect;
import org.spongepowered.asm.mixin.injection.callback.CallbackInfo;

@Mixin(Minecraft.class)
public abstract class MinecraftMixin {
	/** Client tick start: apply held keys/buttons and press edges before handleKeybinds and player input run. */
	@Inject(method = "tick", at = @At("HEAD"))
	private void playerone$tickHead(CallbackInfo ci) {
		InputApplier.onClientTick((Minecraft) (Object) this);
	}

	/** Once per rendered frame, right after vanilla consumed real mouse movement: spread agent camera counts smoothly. */
	@Inject(
		method = "runTick",
		at = @At(value = "INVOKE", target = "Lnet/minecraft/client/MouseHandler;handleAccumulatedMovement()V", shift = At.Shift.AFTER)
	)
	private void playerone$frame(boolean advanceGameTime, CallbackInfo ci) {
		InputApplier.onFrame((Minecraft) (Object) this);
	}

	/** Vanilla opens the pause menu ~500 ms after focus loss; skip that while an agent is in control (options file untouched). */
	@Inject(method = "pauseIfInactive", at = @At("HEAD"), cancellable = true)
	private void playerone$noPauseWhileDriven(CallbackInfo ci) {
		if (InputApplier.isDriving()) {
			ci.cancel();
		}
	}

	/**
	 * handleKeybinds only continues mining (continueAttack) while the mouse is grabbed. Report "grabbed" to that one
	 * check while the bridge is holding attack, without grabbing or hiding the real cursor.
	 */
	@Redirect(
		method = "handleKeybinds",
		at = @At(value = "INVOKE", target = "Lnet/minecraft/client/MouseHandler;isMouseGrabbed()Z")
	)
	private boolean playerone$grabbedForAttack(MouseHandler mouseHandler) {
		return mouseHandler.isMouseGrabbed() || InputApplier.isHoldingAttack((Minecraft) (Object) this);
	}
}
