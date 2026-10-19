package org.playerone.bridge.mixin;

import java.util.function.BooleanSupplier;
import net.minecraft.client.ToggleKeyMapping;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;
import org.spongepowered.asm.mixin.gen.Invoker;

@Mixin(ToggleKeyMapping.class)
public interface ToggleKeyMappingAccessor {
	/** True when the user's options put this mapping in toggle mode (toggle sprint/sneak/attack/use). */
	@Accessor("needsToggle")
	BooleanSupplier playerone$needsToggle();

	/** Forces the mapping up regardless of toggle mode. */
	@Invoker("reset")
	void playerone$reset();
}
