package org.playerone.bridge.mixin;

import net.minecraft.client.KeyMapping;
import org.spongepowered.asm.mixin.Mixin;
import org.spongepowered.asm.mixin.gen.Accessor;

@Mixin(KeyMapping.class)
public interface KeyMappingAccessor {
	@Accessor("clickCount")
	int playerone$getClickCount();

	@Accessor("clickCount")
	void playerone$setClickCount(int value);
}
